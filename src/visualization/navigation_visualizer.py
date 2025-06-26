"""
Navigation visualizer for real-time robot navigation visualization.

Provides real-time visualization of robot movement, obstacles,
goals, and navigation paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from typing import Dict, Any, Optional, List
import time


class NavigationVisualizer:
    """
    Real-time navigation visualizer.
    
    Provides visualization of robot navigation including:
    - Robot position and orientation
    - Obstacles (static and dynamic)
    - Goals and navigation paths
    - Belief states and uncertainty
    """
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100):
        """
        Initialize navigation visualizer.
        
        Args:
            figsize: Figure size
            dpi: DPI for the figure
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Visualization state
        self.robot_trajectory = []
        self.belief_history = []
        self.free_energy_history = []
        
        # Plot elements
        self.robot_plot = None
        self.trajectory_plot = None
        self.obstacle_plots = []
        self.goal_plots = []
        self.belief_plot = None
        
        # Animation
        self.animation = None
        self.is_animating = False
    
    def update(self, environment, agent, belief_data: Optional[Dict] = None):
        """
        Update visualization with current environment and agent state.
        
        Args:
            environment: Navigation environment
            agent: Active inference agent
            belief_data: Optional belief state data
        """
        # Clear previous plots
        self.ax.clear()
        
        # Get environment dimensions
        width = environment.width
        height = environment.height
        
        # Set up plot
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Active Inference Robot Navigation', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Draw obstacles
        self._draw_obstacles(environment)
        
        # Draw goals
        self._draw_goals(environment)
        
        # Draw robot
        self._draw_robot(environment)
        
        # Draw trajectory
        self._draw_trajectory()
        
        # Draw belief state if available
        if belief_data and agent.current_beliefs is not None:
            self._draw_belief_state(agent.current_beliefs)
        
        # Add legend
        self._add_legend()
        
        # Update display
        plt.tight_layout()
        plt.pause(0.01)
    
    def _draw_obstacles(self, environment):
        """Draw obstacles in the environment."""
        # Draw static obstacles
        for obstacle in environment.static_obstacles:
            circle = plt.Circle(
                obstacle.position, 
                obstacle.size, 
                color='gray', 
                alpha=0.7, 
                label='Static Obstacle'
            )
            self.ax.add_patch(circle)
        
        # Draw dynamic obstacles
        for obstacle in environment.dynamic_obstacles:
            circle = plt.Circle(
                obstacle.position, 
                obstacle.size, 
                color='red', 
                alpha=0.7, 
                label='Dynamic Obstacle'
            )
            self.ax.add_patch(circle)
            
            # Draw velocity vector
            if np.any(obstacle.velocity):
                end_pos = obstacle.position + obstacle.velocity * 2
                self.ax.arrow(
                    obstacle.position[0], obstacle.position[1],
                    obstacle.velocity[0] * 2, obstacle.velocity[1] * 2,
                    head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.8
                )
    
    def _draw_goals(self, environment):
        """Draw goals in the environment."""
        for i, goal in enumerate(environment.goals):
            if goal.active:
                # Active goal
                circle = plt.Circle(
                    goal.position, 
                    goal.radius, 
                    color='green', 
                    fill=False, 
                    linewidth=3, 
                    label=f'Goal {i+1}'
                )
                self.ax.add_patch(circle)
                
                # Goal center
                self.ax.plot(goal.position[0], goal.position[1], 'go', markersize=8)
            else:
                # Reached goal
                circle = plt.Circle(
                    goal.position, 
                    goal.radius, 
                    color='darkgreen', 
                    fill=False, 
                    linewidth=2, 
                    alpha=0.5
                )
                self.ax.add_patch(circle)
                self.ax.plot(goal.position[0], goal.position[1], 'go', markersize=6, alpha=0.5)
    
    def _draw_robot(self, environment):
        """Draw robot and its orientation."""
        robot = environment.robot
        
        # Robot position
        self.ax.plot(robot.position[0], robot.position[1], 'bo', markersize=10, label='Robot')
        
        # Robot orientation
        orientation_length = 2.0
        end_x = robot.position[0] + orientation_length * np.cos(robot.orientation)
        end_y = robot.position[1] + orientation_length * np.sin(robot.orientation)
        
        self.ax.arrow(
            robot.position[0], robot.position[1],
            orientation_length * np.cos(robot.orientation),
            orientation_length * np.sin(robot.orientation),
            head_width=0.8, head_length=0.5, fc='blue', ec='blue', alpha=0.8
        )
        
        # Robot velocity
        if np.any(robot.velocity):
            vel_length = np.linalg.norm(robot.velocity)
            if vel_length > 0.1:  # Only show if moving
                self.ax.arrow(
                    robot.position[0], robot.position[1],
                    robot.velocity[0], robot.velocity[1],
                    head_width=0.6, head_length=0.4, fc='cyan', ec='cyan', alpha=0.6
                )
        
        # Store trajectory
        self.robot_trajectory.append(robot.position.copy())
    
    def _draw_trajectory(self):
        """Draw robot trajectory."""
        if len(self.robot_trajectory) > 1:
            trajectory = np.array(self.robot_trajectory)
            self.ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                'b-', alpha=0.6, linewidth=2, label='Trajectory'
            )
    
    def _draw_belief_state(self, belief_data: Dict):
        """Draw belief state visualization."""
        if 'mean' in belief_data and 'logvar' in belief_data:
            belief_mean = belief_data['mean'].flatten()
            belief_uncertainty = np.exp(belief_data['logvar'].flatten())
            
            # Create belief visualization (simplified)
            # This could be enhanced to show more detailed belief representations
            
            # Show belief uncertainty as a circle around robot
            if len(belief_uncertainty) >= 2:
                # Use first two dimensions for position uncertainty
                pos_uncertainty = np.mean(belief_uncertainty[:2])
                robot_pos = self.robot_trajectory[-1] if self.robot_trajectory else [0, 0]
                
                uncertainty_circle = plt.Circle(
                    robot_pos, 
                    pos_uncertainty * 2,  # Scale for visibility
                    color='yellow', 
                    fill=False, 
                    linewidth=2, 
                    alpha=0.7, 
                    label='Belief Uncertainty'
                )
                self.ax.add_patch(uncertainty_circle)
    
    def _add_legend(self):
        """Add legend to the plot."""
        # Get unique labels
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        if by_label:
            self.ax.legend(by_label.values(), by_label.keys(), 
                          loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def render(self):
        """Render the current visualization."""
        plt.draw()
        plt.pause(0.01)
    
    def save_frame(self, filename: str):
        """Save current frame to file."""
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
    
    def create_animation(self, environment, agent, num_frames: int = 100, 
                        interval: int = 100) -> FuncAnimation:
        """
        Create an animation of the navigation.
        
        Args:
            environment: Navigation environment
            agent: Active inference agent
            num_frames: Number of animation frames
            interval: Interval between frames (ms)
            
        Returns:
            Matplotlib animation object
        """
        def animate(frame):
            # Simulate environment step
            if hasattr(agent, 'last_observation') and agent.last_observation is not None:
                action = agent.select_action(agent.last_observation, training=False)
                observation, reward, done, info = environment.step(action)
                
                # Update visualization
                self.update(environment, agent)
                
                if done:
                    environment.reset()
            
            return []
        
        self.animation = FuncAnimation(
            self.fig, animate, frames=num_frames, 
            interval=interval, blit=False, repeat=True
        )
        
        return self.animation
    
    def plot_metrics(self, agent, save_path: Optional[str] = None):
        """
        Plot training metrics.
        
        Args:
            agent: Active inference agent
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training statistics
        stats = agent.get_training_stats()
        
        if stats:
            # Episode rewards (if available)
            if hasattr(agent, 'episode_rewards') and agent.episode_rewards:
                axes[0, 0].plot(agent.episode_rewards)
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True)
            
            # Free energy history
            if agent.free_energy_history:
                recent_fe = agent.free_energy_history[-100:]
                axes[0, 1].plot(recent_fe)
                axes[0, 1].set_title('Variational Free Energy')
                axes[0, 1].set_xlabel('Update Step')
                axes[0, 1].set_ylabel('Free Energy')
                axes[0, 1].grid(True)
            
            # Training losses
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
            
            # Exploration rate
            if 'exploration_rate' in stats:
                axes[1, 1].plot([stats['exploration_rate']])
                axes[1, 1].set_title('Exploration Rate')
                axes[1, 1].set_xlabel('Current')
                axes[1, 1].set_ylabel('Rate')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def clear_trajectory(self):
        """Clear the robot trajectory."""
        self.robot_trajectory = []
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig) 