#!/usr/bin/env python3
"""
Curriculum Learning Visualization Dashboard
Shows curriculum progression, success rates, and waypoint usage over time.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
from collections import defaultdict
import seaborn as sns

def load_experiment_data():
    """Load experiment data from CSV files."""
    files = glob.glob('data/experiments/step_log_episode_*.csv')
    files.sort(key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
    
    episodes = []
    successes = []
    steps = []
    rewards = []
    collisions = []
    
    for file in files:
        episode_num = int(re.search(r'episode_(\d+)', file).group(1))
        df = pd.read_csv(file)
        
        episodes.append(episode_num)
        successes.append(df['goal_reached'].iloc[-1])
        steps.append(len(df))
        rewards.append(df['reward'].sum())
        collisions.append(df['collision'].sum())
    
    return pd.DataFrame({
        'episode': episodes,
        'success': successes,
        'steps': steps,
        'total_reward': rewards,
        'collisions': collisions
    })

def create_curriculum_dashboard():
    """Create a comprehensive curriculum learning dashboard."""
    df = load_experiment_data()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curriculum Learning Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Success Rate Over Time
    window_size = 10
    moving_avg = df['success'].rolling(window=window_size).mean()
    axes[0, 0].plot(df['episode'], df['success'], alpha=0.3, color='blue', label='Raw Success')
    axes[0, 0].plot(df['episode'], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    axes[0, 0].set_title('Success Rate Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Steps per Episode
    axes[0, 1].scatter(df['episode'], df['steps'], alpha=0.5, s=10)
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Total Reward per Episode
    axes[0, 2].scatter(df['episode'], df['total_reward'], alpha=0.5, s=10, color='green')
    axes[0, 2].set_title('Total Reward per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Total Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Collisions per Episode
    axes[1, 0].scatter(df['episode'], df['collisions'], alpha=0.5, s=10, color='red')
    axes[1, 0].set_title('Collisions per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Collisions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Success Rate Distribution
    successful_episodes = df[df['success'] == True]
    failed_episodes = df[df['success'] == False]
    
    axes[1, 1].hist([successful_episodes['steps'], failed_episodes['steps']], 
                    bins=20, alpha=0.7, label=['Successful', 'Failed'])
    axes[1, 1].set_title('Step Distribution by Success')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance Metrics Summary
    total_episodes = len(df)
    successful_episodes = df['success'].sum()
    success_rate = successful_episodes / total_episodes * 100
    avg_steps = df['steps'].mean()
    avg_reward = df['total_reward'].mean()
    total_collisions = df['collisions'].sum()
    
    summary_text = f"""
    Total Episodes: {total_episodes}
    Successful Episodes: {successful_episodes}
    Success Rate: {success_rate:.1f}%
    Average Steps: {avg_steps:.1f}
    Average Reward: {avg_reward:.1f}
    Total Collisions: {total_collisions}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('curriculum_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_curriculum_progression_plot():
    """Create a plot showing curriculum progression over episodes."""
    df = load_experiment_data()
    
    # Simulate curriculum progression based on episode numbers
    # In a real implementation, you'd track this from the environment
    curriculum_levels = []
    for episode in df['episode']:
        if episode <= 50:
            level = 0
        elif episode <= 100:
            level = 1
        elif episode <= 150:
            level = 2
        elif episode <= 200:
            level = 3
        else:
            level = 4
        curriculum_levels.append(level)
    
    df['curriculum_level'] = curriculum_levels
    
    plt.figure(figsize=(12, 8))
    
    # Plot curriculum level over time
    plt.subplot(2, 1, 1)
    plt.plot(df['episode'], df['curriculum_level'], 'o-', linewidth=2, markersize=4)
    plt.title('Curriculum Level Progression')
    plt.xlabel('Episode')
    plt.ylabel('Curriculum Level')
    plt.grid(True, alpha=0.3)
    
    # Plot success rate with curriculum level overlay
    plt.subplot(2, 1, 2)
    window_size = 10
    moving_avg = df['success'].rolling(window=window_size).mean()
    
    # Color code by curriculum level
    for level in df['curriculum_level'].unique():
        mask = df['curriculum_level'] == level
        plt.scatter(df[mask]['episode'], moving_avg[mask], 
                   label=f'Level {level}', alpha=0.7, s=20)
    
    plt.title('Success Rate by Curriculum Level')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curriculum_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating Curriculum Learning Dashboard...")
    df = create_curriculum_dashboard()
    
    print("Creating Curriculum Progression Plot...")
    create_curriculum_progression_plot()
    
    print("Visualization complete! Check the generated PNG files.") 