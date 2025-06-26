"""
Navigation environment for adaptive robot navigation.

Implements a 2D grid world with dynamic obstacles, changing goals,
and multiple sensor modalities for testing active inference agents.
"""

import numpy as np
import yaml
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
import cv2
import gym
from .sensors import LIDARSensor, CameraSensor, IMUSensor


@dataclass
class RobotState:
    """Robot state representation."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    orientation: float    # radians
    battery: float        # battery level


@dataclass
class Goal:
    """Goal representation."""
    position: np.ndarray  # [x, y]
    radius: float
    reward: float
    active: bool = True


@dataclass
class Obstacle:
    """Obstacle representation."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy] (for dynamic obstacles)
    size: float
    dynamic: bool = False


class NavigationEnvironment:
    """
    Navigation environment with dynamic obstacles and goals.
    
    Features:
    - 2D grid world with continuous robot movement
    - Static and dynamic obstacles
    - Multiple goals that can change during episodes
    - Multiple sensor modalities (LIDAR, camera, IMU)
    - Collision detection and physics
    """
    
    def __init__(self, config_path: str = "config/environment_config.yaml"):
        """Initialize the navigation environment."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['environment']
        
        # Environment dimensions
        self.width = self.config['width']
        self.height = self.config['height']
        
        # Robot parameters
        self.robot_config = self.config['robot']
        self.max_speed = self.robot_config['max_speed']
        self.sensor_range = self.robot_config['sensor_range']
        
        # Initialize robot state
        self.robot = RobotState(
            position=np.array(self.robot_config['initial_position'], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            orientation=0.0,
            battery=100.0
        )
        
        # Initialize sensors
        self._init_sensors()
        
        # Initialize environment elements
        self.goals: List[Goal] = []
        self.static_obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[Obstacle] = []
        
        # Environment state
        self.step_count = 0
        self.max_steps = self.config['physics']['max_steps']
        self.current_goal_idx = 0
        
        # Action space: [forward, backward, left, right]
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space (will be set after sensor initialization)
        self.observation_space = None
        self._set_observation_space()
        
        # Initialize environment
        self.reset()
    
    def _init_sensors(self):
        """Initialize sensor systems."""
        sensor_config = self.config['sensors']
        
        # LIDAR sensor
        lidar_config = sensor_config['lidar']
        self.lidar = LIDARSensor(
            num_rays=lidar_config['num_rays'],
            max_range=lidar_config['max_range'],
            noise_std=lidar_config['noise_std']
        )
        
        # Camera sensor
        camera_config = sensor_config['camera']
        self.camera = CameraSensor(
            resolution=camera_config['resolution'],
            field_of_view=camera_config['field_of_view'],
            noise_std=camera_config['noise_std']
        )
        
        # IMU sensor
        imu_config = sensor_config['imu']
        self.imu = IMUSensor(noise_std=imu_config['noise_std'])
    
    def _set_observation_space(self):
        """Set the observation space based on sensor configurations."""
        # LIDAR observations
        lidar_dim = self.lidar.num_rays
        
        # Camera observations (grayscale)
        camera_dim = self.camera.resolution[0] * self.camera.resolution[1]
        
        # IMU observations (position, velocity, orientation)
        imu_dim = 6
        
        # Goal information
        goal_dim = 4  # [goal_x, goal_y, distance_to_goal, angle_to_goal]
        
        total_dim = lidar_dim + camera_dim + imu_dim + goal_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_dim,), 
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reset robot state
        self.robot.position = np.array(self.robot_config['initial_position'], dtype=np.float32)
        self.robot.velocity = np.zeros(2, dtype=np.float32)
        self.robot.orientation = 0.0
        self.robot.battery = 100.0
        
        # Reset environment state
        self.step_count = 0
        self.current_goal_idx = 0
        
        # Generate new environment layout
        self._generate_obstacles()
        self._generate_goals()
        
        # Get initial observation
        observation = self._get_observation()
        return observation
    
    def _generate_obstacles(self):
        """Generate static and dynamic obstacles."""
        obstacle_config = self.config['obstacles']
        
        # Clear existing obstacles
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        
        # Generate static obstacles
        for _ in range(obstacle_config['num_static']):
            obstacle = self._generate_random_obstacle(dynamic=False)
            self.static_obstacles.append(obstacle)
        
        # Generate dynamic obstacles
        for _ in range(obstacle_config['num_dynamic']):
            obstacle = self._generate_random_obstacle(dynamic=True)
            self.dynamic_obstacles.append(obstacle)
    
    def _generate_random_obstacle(self, dynamic: bool = False) -> Obstacle:
        """Generate a random obstacle."""
        # Generate position away from robot
        while True:
            position = np.random.uniform(
                [0, 0], 
                [self.width, self.height]
            )
            
            # Check distance from robot
            distance = np.linalg.norm(position - self.robot.position)
            if distance > 5.0:  # Minimum distance from robot
                break
        
        # Generate velocity for dynamic obstacles
        if dynamic:
            speed = self.config['obstacles']['dynamic_speed']
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            velocity = np.zeros(2)
        
        return Obstacle(
            position=position,
            velocity=velocity,
            size=self.config['obstacles']['obstacle_size'],
            dynamic=dynamic
        )
    
    def _generate_goals(self):
        """Generate goals for the episode."""
        goal_config = self.config['goals']
        
        self.goals.clear()
        for _ in range(goal_config['num_goals']):
            # Generate goal position away from obstacles
            while True:
                position = np.random.uniform(
                    [0, 0], 
                    [self.width, self.height]
                )
                
                # Check distance from robot
                distance = np.linalg.norm(position - self.robot.position)
                if distance > 10.0:  # Minimum distance from robot
                    break
            
            goal = Goal(
                position=position,
                radius=goal_config['goal_radius'],
                reward=goal_config['goal_reward']
            )
            self.goals.append(goal)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0: forward, 1: backward, 2: left, 3: right)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Update step count
        self.step_count += 1
        
        # Apply action to robot
        self._apply_action(action)
        
        # Update dynamic obstacles
        if self.config['dynamics']['obstacle_movement']:
            self._update_dynamic_obstacles()
        
        # Update goals (change current goal periodically)
        if self.config['dynamics']['goal_changes']:
            self._update_goals()
        
        # Check collisions
        collision = self._check_collisions()
        
        # Check goal reached
        goal_reached = self._check_goal_reached()
        
        # Calculate reward
        reward = self._calculate_reward(collision, goal_reached)
        
        # Check if episode is done
        done = self._is_done(collision, goal_reached)
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info
        info = {
            'collision': collision,
            'goal_reached': goal_reached,
            'current_goal': self.current_goal_idx,
            'robot_position': self.robot.position.copy(),
            'robot_orientation': self.robot.orientation,
            'step_count': self.step_count
        }
        
        return observation, reward, done, info
    
    def _apply_action(self, action: int):
        """Apply action to robot."""
        # Action mapping: 0: forward, 1: backward, 2: left, 3: right
        action_angles = [0, np.pi, np.pi/2, -np.pi/2]
        angle = action_angles[action]
        
        # Calculate new velocity
        new_velocity = self.max_speed * np.array([
            np.cos(self.robot.orientation + angle),
            np.sin(self.robot.orientation + angle)
        ])
        
        # Update robot state
        self.robot.velocity = new_velocity
        self.robot.position += new_velocity
        
        # Update orientation (for IMU)
        self.robot.orientation += angle * 0.1  # Small orientation change
        
        # Keep robot within bounds
        self.robot.position = np.clip(
            self.robot.position, 
            [0, 0], 
            [self.width, self.height]
        )
        
        # Update battery
        self.robot.battery -= 0.1
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles."""
        for obstacle in self.dynamic_obstacles:
            # Update position
            obstacle.position += obstacle.velocity
            
            # Bounce off boundaries
            for i in range(2):
                if obstacle.position[i] <= 0 or obstacle.position[i] >= [self.width, self.height][i]:
                    obstacle.velocity[i] *= -1
            
            # Keep within bounds
            obstacle.position = np.clip(
                obstacle.position,
                [0, 0],
                [self.width, self.height]
            )
    
    def _update_goals(self):
        """Update goals (change current goal periodically)."""
        change_freq = self.config['dynamics']['change_frequency']
        if self.step_count % change_freq == 0 and len(self.goals) > 1:
            # Change to next goal
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.goals)
    
    def _check_collisions(self) -> bool:
        """Check for collisions with obstacles."""
        for obstacle in self.static_obstacles + self.dynamic_obstacles:
            distance = np.linalg.norm(
                self.robot.position - obstacle.position
            )
            if distance < obstacle.size:
                return True
        return False
    
    def _check_goal_reached(self) -> bool:
        """Check if current goal has been reached."""
        if not self.goals or self.current_goal_idx >= len(self.goals):
            return False
        
        current_goal = self.goals[self.current_goal_idx]
        if not current_goal.active:
            return False
        
        distance = np.linalg.norm(
            self.robot.position - current_goal.position
        )
        return distance <= current_goal.radius
    
    def _calculate_reward(self, collision: bool, goal_reached: bool) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        
        # Step penalty
        reward += self.config['physics']['step_penalty']
        
        # Collision penalty
        if collision:
            reward += self.config['physics']['collision_penalty']
        
        # Goal reward
        if goal_reached:
            current_goal = self.goals[self.current_goal_idx]
            reward += current_goal.reward
            current_goal.active = False  # Deactivate reached goal
        
        # Distance-based reward (closer to goal = higher reward)
        if self.goals and self.current_goal_idx < len(self.goals):
            current_goal = self.goals[self.current_goal_idx]
            if current_goal.active:
                distance = np.linalg.norm(
                    self.robot.position - current_goal.position
                )
                reward += 1.0 / (1.0 + distance)  # Closer = higher reward
        
        return reward
    
    def _is_done(self, collision: bool, goal_reached: bool) -> bool:
        """Check if episode is done."""
        # Time limit reached
        if self.step_count >= self.max_steps:
            return True
        
        # All goals reached
        if all(not goal.active for goal in self.goals):
            return True
        
        # Fatal collision
        if collision:
            return True
        
        # Battery depleted
        if self.robot.battery <= 0:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from all sensors."""
        observations = []
        
        # LIDAR observation
        lidar_obs = self.lidar.get_observation(
            self.robot.position,
            self.robot.orientation,
            self.static_obstacles + self.dynamic_obstacles
        )
        observations.append(lidar_obs)
        
        # Camera observation
        camera_obs = self.camera.get_observation(
            self.robot.position,
            self.robot.orientation,
            self.static_obstacles + self.dynamic_obstacles,
            self.goals
        )
        observations.append(camera_obs.flatten())
        
        # IMU observation
        imu_obs = self.imu.get_observation(
            self.robot.position,
            self.robot.velocity,
            self.robot.orientation
        )
        observations.append(imu_obs)
        
        # Goal information
        goal_obs = self._get_goal_observation()
        observations.append(goal_obs)
        
        return np.concatenate(observations, dtype=np.float32)
    
    def _get_goal_observation(self) -> np.ndarray:
        """Get goal-related observation."""
        if not self.goals or self.current_goal_idx >= len(self.goals):
            return np.zeros(4, dtype=np.float32)
        
        current_goal = self.goals[self.current_goal_idx]
        goal_pos = current_goal.position
        
        # Distance to goal
        distance = np.linalg.norm(self.robot.position - goal_pos)
        
        # Angle to goal
        angle = np.arctan2(
            goal_pos[1] - self.robot.position[1],
            goal_pos[0] - self.robot.position[0]
        ) - self.robot.orientation
        
        # Normalize angle to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        return np.array([
            goal_pos[0], goal_pos[1], distance, angle
        ], dtype=np.float32)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        # Create visualization
        img = np.ones((self.height * 10, self.width * 10, 3), dtype=np.uint8) * 255
        
        # Draw obstacles
        for obstacle in self.static_obstacles:
            pos = (obstacle.position * 10).astype(int)
            cv2.circle(img, tuple(pos), int(obstacle.size * 10), (100, 100, 100), -1)
        
        for obstacle in self.dynamic_obstacles:
            pos = (obstacle.position * 10).astype(int)
            cv2.circle(img, tuple(pos), int(obstacle.size * 10), (200, 100, 100), -1)
        
        # Draw goals
        for i, goal in enumerate(self.goals):
            pos = (goal.position * 10).astype(int)
            color = (0, 255, 0) if i == self.current_goal_idx and goal.active else (0, 200, 0)
            cv2.circle(img, tuple(pos), int(goal.radius * 10), color, 2)
        
        # Draw robot
        robot_pos = (self.robot.position * 10).astype(int)
        cv2.circle(img, tuple(robot_pos), 5, (0, 0, 255), -1)
        
        # Draw robot orientation
        end_pos = robot_pos + (5 * np.array([
            np.cos(self.robot.orientation),
            np.sin(self.robot.orientation)
        ])).astype(int)
        cv2.line(img, tuple(robot_pos), tuple(end_pos), (0, 0, 255), 2)
        
        if mode == 'human':
            cv2.imshow('Navigation Environment', img)
            cv2.waitKey(1)
            return None
        else:
            return img 