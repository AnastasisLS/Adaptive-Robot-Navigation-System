#!/usr/bin/env python3
"""
Analysis script for Adaptive Robot Navigation experiment results
"""

import pandas as pd
import numpy as np
import glob
import re
from collections import defaultdict

def analyze_results():
    # Get all episode files
    files = glob.glob('data/experiments/step_log_episode_*.csv')
    files.sort(key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
    
    episodes = []
    successes = []
    steps = []
    rewards = []
    collisions = []
    
    print("Analyzing experiment results...")
    
    for file in files:
        episode_num = int(re.search(r'episode_(\d+)', file).group(1))
        df = pd.read_csv(file)
        
        episodes.append(episode_num)
        successes.append(df['goal_reached'].iloc[-1])
        steps.append(len(df))
        rewards.append(df['reward'].sum())
        collisions.append(df['collision'].sum())
    
    # Calculate overall statistics
    total_episodes = len(episodes)
    successful_episodes = sum(successes)
    success_rate = successful_episodes / total_episodes * 100
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Failed Episodes: {total_episodes - successful_episodes}")
    print(f"Overall Success Rate: {success_rate:.1f}%")
    print(f"Average Steps per Episode: {np.mean(steps):.1f}")
    print(f"Average Reward per Episode: {np.mean(rewards):.1f}")
    print(f"Total Collisions: {sum(collisions)}")
    print(f"Average Collisions per Episode: {np.mean(collisions):.2f}")
    
    # Curriculum phase analysis
    curriculum_phases = {
        'Episodes 1-8 (1 static)': (1, 8),
        'Episodes 9-13 (2 static)': (9, 13),
        'Episodes 14-15 (3 static)': (14, 15),
        'Episodes 16-18 (3 static + 1 dynamic)': (16, 18),
        'Episodes 19-25 (4-5 static + 1 dynamic)': (19, 25),
        'Episodes 26-30 (5-6 static + 1 dynamic)': (26, 30),
        'Episodes 31-32 (6 static + 1 dynamic)': (31, 32),
        'Episodes 33 (6 static + 2 dynamic)': (33, 33),
        'Episodes 34-39 (7-8 static + 2 dynamic)': (34, 39),
        'Episodes 40-48 (8-9 static + 2 dynamic)': (40, 48),
        'Episodes 49-133 (10 static + 3 dynamic)': (49, 133)
    }
    
    print(f"\n{'='*60}")
    print(f"CURRICULUM PHASE ANALYSIS")
    print(f"{'='*60}")
    
    for phase_name, (start, end) in curriculum_phases.items():
        phase_episodes = [i for i in episodes if start <= i <= end]
        if phase_episodes:
            phase_successes = sum([successes[episodes.index(i)] for i in phase_episodes])
            phase_rate = phase_successes / len(phase_episodes) * 100
            phase_steps = [steps[episodes.index(i)] for i in phase_episodes]
            phase_rewards = [rewards[episodes.index(i)] for i in phase_episodes]
            
            print(f"{phase_name}:")
            print(f"  Success Rate: {phase_successes}/{len(phase_episodes)} = {phase_rate:.1f}%")
            print(f"  Avg Steps: {np.mean(phase_steps):.1f}")
            print(f"  Avg Reward: {np.mean(phase_rewards):.1f}")
            print()
    
    # Learning progression analysis
    print(f"{'='*60}")
    print(f"LEARNING PROGRESSION ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate moving average success rate
    window_size = 10
    moving_avg = []
    for i in range(window_size, len(successes)):
        window_success_rate = sum(successes[i-window_size:i]) / window_size * 100
        moving_avg.append(window_success_rate)
    
    if moving_avg:
        print(f"Moving Average Success Rate (window={window_size}):")
        print(f"  First 10 episodes: {moving_avg[0]:.1f}%")
        print(f"  Last 10 episodes: {moving_avg[-1]:.1f}%")
        print(f"  Overall trend: {'Improving' if moving_avg[-1] > moving_avg[0] else 'Declining'}")
    
    # Performance metrics
    successful_steps = [steps[i] for i in range(len(steps)) if successes[i]]
    failed_steps = [steps[i] for i in range(len(steps)) if not successes[i]]
    
    if successful_steps:
        print(f"\nSuccessful Episodes:")
        print(f"  Average steps: {np.mean(successful_steps):.1f}")
        print(f"  Min steps: {min(successful_steps)}")
        print(f"  Max steps: {max(successful_steps)}")
    
    if failed_steps:
        print(f"\nFailed Episodes:")
        print(f"  Average steps: {np.mean(failed_steps):.1f}")
        print(f"  Min steps: {min(failed_steps)}")
        print(f"  Max steps: {max(failed_steps)}")
    
    print(f"\n{'='*60}")
    print(f"KEY INSIGHTS")
    print(f"{'='*60}")
    
    # Key insights
    if success_rate > 80:
        print("✅ EXCELLENT: Very high success rate indicates effective learning")
    elif success_rate > 60:
        print("✅ GOOD: Solid success rate shows the agent is learning effectively")
    elif success_rate > 40:
        print("⚠️  MODERATE: Moderate success rate suggests learning is occurring but challenging")
    else:
        print("❌ CHALLENGING: Low success rate indicates the task is very difficult")
    
    if np.mean(collisions) < 0.5:
        print("✅ SAFE: Very few collisions, excellent obstacle avoidance")
    elif np.mean(collisions) < 1.0:
        print("✅ GOOD: Low collision rate, good obstacle avoidance")
    else:
        print("⚠️  COLLISIONS: Higher collision rate, room for improvement")
    
    if successful_steps and np.mean(successful_steps) < 100:
        print("✅ EFFICIENT: Successful episodes are very fast")
    elif successful_steps and np.mean(successful_steps) < 200:
        print("✅ GOOD: Reasonable efficiency in successful episodes")
    elif successful_steps:
        print("⚠️  SLOW: Successful episodes take many steps")
    
    print(f"\nExperiment Status: {'COMPLETED' if total_episodes >= 500 else 'IN PROGRESS'}")
    if total_episodes < 500:
        remaining = 500 - total_episodes
        print(f"Remaining Episodes: {remaining}")
        print(f"Estimated Time Remaining: {remaining * 4.4 / 60:.1f} hours")

if __name__ == "__main__":
    analyze_results() 