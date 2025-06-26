#!/usr/bin/env python3
"""
Basic navigation example using active inference.

Demonstrates the active inference agent navigating to goals
in a simple environment with static obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from src.environment import NavigationEnvironment
from src.agents import ActiveInferenceAgent
from src.visualization import NavigationVisualizer


def run_basic_navigation(num_episodes: int = 10, max_steps: int = 1000):
    """
    Run basic navigation experiment.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("Setting up navigation environment...")
    
    # Create environment
    env = NavigationEnvironment()
    
    # Create active inference agent
    agent = ActiveInferenceAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    # Create visualizer
    visualizer = NavigationVisualizer()
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Starting {num_episodes} episodes of basic navigation...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        
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
            
            # Visualize (every 10 steps to avoid slowing down)
            if step % 10 == 0:
                visualizer.update(env, agent)
                visualizer.render()
            
            # Check if episode is done
            if done:
                if info.get('goal_reached', False):
                    success_count += 1
                    print(f"  ✓ Goal reached in {episode_length} steps")
                else:
                    print(f"  ✗ Episode failed after {episode_length} steps")
                break
            
            observation = next_observation
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print episode summary
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        
        # Get training statistics
        stats = agent.get_training_stats()
        if stats:
            print(f"  Avg Free Energy: {stats.get('avg_free_energy', 'N/A'):.4f}")
            print(f"  Exploration Rate: {stats.get('exploration_rate', 'N/A'):.3f}")
    
    # Print final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, agent)
    
    # Save trained model
    agent.save_models("data/models/active_inference_basic_navigation.pth")
    print("Model saved to data/models/active_inference_basic_navigation.pth")
    
    return agent, env


def plot_training_curves(rewards, lengths, agent):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Training losses (if available)
    if agent.training_losses:
        recent_losses = agent.training_losses[-100:]
        gen_losses = [l.get('generative_loss', 0) for l in recent_losses]
        recog_losses = [l.get('recognition_loss', 0) for l in recent_losses]
        policy_losses = [l.get('policy_loss', 0) for l in recent_losses]
        
        axes[1, 0].plot(gen_losses, label='Generative')
        axes[1, 0].plot(recog_losses, label='Recognition')
        axes[1, 0].plot(policy_losses, label='Policy')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Free energy history
    if agent.free_energy_history:
        recent_fe = agent.free_energy_history[-100:]
        axes[1, 1].plot(recent_fe)
        axes[1, 1].set_title('Variational Free Energy')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Free Energy')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('data/experiments/basic_navigation_training.png', dpi=300, bbox_inches='tight')
    print('Training curves saved to data/experiments/basic_navigation_training.png')
    plt.close(fig)


def demonstrate_agent(agent, env, num_demonstrations: int = 3):
    """
    Demonstrate the trained agent.
    
    Args:
        agent: Trained agent
        env: Environment
        num_demonstrations: Number of demonstrations to show
    """
    print(f"\n=== Demonstrating Trained Agent ===")
    
    visualizer = NavigationVisualizer()
    
    for demo in range(num_demonstrations):
        print(f"\nDemonstration {demo + 1}/{num_demonstrations}")
        
        observation = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 500:
            # Select action (no exploration)
            action = agent.select_action(observation, training=False)
            
            # Take step
            observation, reward, done, info = env.step(action)
            
            # Visualize
            visualizer.update(env, agent)
            visualizer.render()
            
            step_count += 1
            time.sleep(0.1)  # Slow down for visualization
        
        if info.get('goal_reached', False):
            print(f"  ✓ Demonstration successful!")
        else:
            print(f"  ✗ Demonstration failed")


if __name__ == "__main__":
    # Create data directories
    import os
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/experiments", exist_ok=True)
    
    # Run basic navigation experiment
    agent, env = run_basic_navigation(num_episodes=20, max_steps=500)
    
    # Demonstrate trained agent
    demonstrate_agent(agent, env, num_demonstrations=3)
    
    print("\nBasic navigation experiment completed!") 