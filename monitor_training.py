#!/usr/bin/env python3
"""
Real-time Training Monitor
Shows live curriculum progress and metrics during training.
"""

import time
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class TrainingMonitor:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16)
        
    def get_latest_data(self):
        """Get the latest experiment data."""
        files = glob.glob('data/experiments/step_log_episode_*.csv')
        if not files:
            return pd.DataFrame()
        
        files.sort(key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
        
        episodes = []
        successes = []
        steps = []
        rewards = []
        
        for file in files:
            episode_num = int(re.search(r'episode_(\d+)', file).group(1))
            df = pd.read_csv(file)
            
            episodes.append(episode_num)
            successes.append(df['goal_reached'].iloc[-1])
            steps.append(len(df))
            rewards.append(df['reward'].sum())
        
        return pd.DataFrame({
            'episode': episodes,
            'success': successes,
            'steps': steps,
            'total_reward': rewards
        })
    
    def update_plots(self, frame):
        """Update all plots with latest data."""
        df = self.get_latest_data()
        if df.empty:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. Success Rate Over Time
        window_size = min(10, len(df))
        if window_size > 0:
            moving_avg = df['success'].rolling(window=window_size).mean()
            self.axes[0, 0].plot(df['episode'], df['success'], alpha=0.3, color='blue', label='Raw Success')
            self.axes[0, 0].plot(df['episode'], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            self.axes[0, 0].set_title('Success Rate Over Time')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Success Rate')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Steps per Episode
        self.axes[0, 1].scatter(df['episode'], df['steps'], alpha=0.5, s=10)
        self.axes[0, 1].set_title('Steps per Episode')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total Reward per Episode
        self.axes[1, 0].scatter(df['episode'], df['total_reward'], alpha=0.5, s=10, color='green')
        self.axes[1, 0].set_title('Total Reward per Episode')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Total Reward')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance Summary
        total_episodes = len(df)
        successful_episodes = df['success'].sum()
        success_rate = successful_episodes / total_episodes * 100 if total_episodes > 0 else 0
        avg_steps = df['steps'].mean() if len(df) > 0 else 0
        avg_reward = df['total_reward'].mean() if len(df) > 0 else 0
        
        summary_text = f"""
        Total Episodes: {total_episodes}
        Successful: {successful_episodes}
        Success Rate: {success_rate:.1f}%
        Avg Steps: {avg_steps:.1f}
        Avg Reward: {avg_reward:.1f}
        """
        
        self.axes[1, 1].text(0.1, 0.5, summary_text, transform=self.axes[1, 1].transAxes,
                             fontsize=12, verticalalignment='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.axes[1, 1].set_title('Live Performance Summary')
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        print("Starting real-time training monitor...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            ani = FuncAnimation(self.fig, self.update_plots, interval=5000)  # Update every 5 seconds
            plt.show()
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

def main():
    monitor = TrainingMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main() 