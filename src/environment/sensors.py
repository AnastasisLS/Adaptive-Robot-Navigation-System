"""
Sensor implementations for the navigation environment.

Provides LIDAR, camera, and IMU sensor models with realistic noise
and measurement characteristics.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class Obstacle:
    """Obstacle representation for sensors."""
    position: np.ndarray
    velocity: np.ndarray
    size: float
    dynamic: bool = False


@dataclass
class Goal:
    """Goal representation for sensors."""
    position: np.ndarray
    radius: float
    reward: float
    active: bool = True


class LIDARSensor:
    """
    LIDAR sensor simulation.
    
    Simulates a 2D LIDAR sensor with configurable number of rays,
    maximum range, and noise characteristics.
    """
    
    def __init__(self, num_rays: int = 36, max_range: float = 10.0, noise_std: float = 0.1):
        """
        Initialize LIDAR sensor.
        
        Args:
            num_rays: Number of LIDAR rays (angular resolution)
            max_range: Maximum detection range
            noise_std: Standard deviation of measurement noise
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.noise_std = noise_std
        
        # Calculate ray angles (evenly distributed around 360 degrees)
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    def get_observation(self, robot_position: np.ndarray, robot_orientation: float, 
                       obstacles: List[Obstacle]) -> np.ndarray:
        """
        Get LIDAR observation.
        
        Args:
            robot_position: Current robot position [x, y]
            robot_orientation: Current robot orientation (radians)
            obstacles: List of obstacles in the environment
            
        Returns:
            Array of distances for each ray
        """
        distances = np.full(self.num_rays, self.max_range)
        
        # Calculate ray directions in world coordinates
        world_angles = self.ray_angles + robot_orientation
        ray_directions = np.column_stack([
            np.cos(world_angles),
            np.sin(world_angles)
        ])
        
        # Check intersection with each obstacle
        for obstacle in obstacles:
            self._check_obstacle_intersection(
                robot_position, ray_directions, obstacle, distances
            )
        
        # Add noise to measurements
        noise = np.random.normal(0, self.noise_std, self.num_rays)
        distances += noise
        
        # Clip to valid range
        distances = np.clip(distances, 0, self.max_range)
        
        return distances.astype(np.float32)
    
    def get_readings(self, robot_position: np.ndarray, robot_orientation: float, 
                    obstacles: List[Obstacle]) -> np.ndarray:
        """Alias for get_observation for backward compatibility."""
        return self.get_observation(robot_position, robot_orientation, obstacles)
    
    def _check_obstacle_intersection(self, robot_position: np.ndarray, 
                                   ray_directions: np.ndarray, obstacle: Obstacle, 
                                   distances: np.ndarray):
        """Check intersection of rays with a single obstacle."""
        # Vector from robot to obstacle center
        to_obstacle = obstacle.position - robot_position
        obstacle_distance = np.linalg.norm(to_obstacle)
        
        # Skip if obstacle is too far
        if obstacle_distance > self.max_range + obstacle.size:
            return
        
        # Calculate angle to obstacle center
        angle_to_obstacle = np.arctan2(to_obstacle[1], to_obstacle[0])
        
        # Check each ray
        for i, ray_dir in enumerate(ray_directions):
            ray_angle = np.arctan2(ray_dir[1], ray_dir[0])
            
            # Calculate angle difference
            angle_diff = np.abs(ray_angle - angle_to_obstacle)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            
            # Check if ray intersects with obstacle
            if angle_diff <= np.arcsin(obstacle.size / obstacle_distance):
                # Calculate intersection distance
                # Using the law of cosines
                cos_angle = np.cos(angle_diff)
                intersection_distance = (obstacle_distance * cos_angle - 
                                       np.sqrt(obstacle.size**2 - 
                                              (obstacle_distance * np.sin(angle_diff))**2))
                
                # Update if this is the closest intersection
                if 0 < intersection_distance < distances[i]:
                    distances[i] = intersection_distance


class CameraSensor:
    """
    Camera sensor simulation.
    
    Simulates a 2D camera sensor with configurable resolution,
    field of view, and noise characteristics.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (64, 64), 
                 field_of_view: float = 90.0, noise_std: float = 0.05):
        """
        Initialize camera sensor.
        
        Args:
            resolution: Camera resolution (width, height)
            field_of_view: Field of view in degrees
            noise_std: Standard deviation of pixel noise
        """
        self.resolution = resolution
        self.field_of_view = np.radians(field_of_view)
        self.noise_std = noise_std
        
        # Calculate pixel coordinates
        self.pixel_x = np.arange(resolution[0])
        self.pixel_y = np.arange(resolution[1])
        self.pixel_xx, self.pixel_yy = np.meshgrid(self.pixel_x, self.pixel_y)
    
    def get_observation(self, robot_position: np.ndarray, robot_orientation: float,
                       obstacles: List[Obstacle], goals: List[Goal]) -> np.ndarray:
        """
        Get camera observation.
        
        Args:
            robot_position: Current robot position [x, y]
            robot_orientation: Current robot orientation (radians)
            obstacles: List of obstacles in the environment
            goals: List of goals in the environment
            
        Returns:
            Grayscale image as 2D array
        """
        # Initialize image
        image = np.zeros(self.resolution, dtype=np.float32)
        
        # Convert world coordinates to camera coordinates
        for obstacle in obstacles:
            self._project_obstacle(robot_position, robot_orientation, obstacle, image)
        
        for goal in goals:
            if goal.active:
                self._project_goal(robot_position, robot_orientation, goal, image)
        
        # Add noise
        noise = np.random.normal(0, self.noise_std, self.resolution)
        image += noise
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def get_image(self, robot_position: np.ndarray, robot_orientation: float,
                  obstacles: List[Obstacle]) -> np.ndarray:
        """Alias for get_observation for backward compatibility."""
        goals = []  # Empty goals list for backward compatibility
        return self.get_observation(robot_position, robot_orientation, obstacles, goals)
    
    def _project_obstacle(self, robot_position: np.ndarray, robot_orientation: float,
                         obstacle: Obstacle, image: np.ndarray):
        """Project an obstacle onto the camera image."""
        # Transform obstacle position to camera coordinates
        world_pos = obstacle.position - robot_position
        
        # Rotate to camera frame
        cos_rot = np.cos(-robot_orientation)
        sin_rot = np.sin(-robot_orientation)
        camera_x = world_pos[0] * cos_rot - world_pos[1] * sin_rot
        camera_y = world_pos[0] * sin_rot + world_pos[1] * cos_rot
        
        # Skip if behind camera
        if camera_x <= 0:
            return
        
        # Project to image coordinates
        # Using perspective projection
        focal_length = self.resolution[0] / (2 * np.tan(self.field_of_view / 2))
        
        pixel_x = int(focal_length * camera_y / camera_x + self.resolution[0] / 2)
        pixel_y = int(focal_length * camera_x / camera_x + self.resolution[1] / 2)
        
        # Check if in image bounds
        if (0 <= pixel_x < self.resolution[0] and 
            0 <= pixel_y < self.resolution[1]):
            # Draw obstacle as a circle
            radius = int(obstacle.size * focal_length / camera_x)
            cv2.circle(image, (pixel_x, pixel_y), radius, 0.8, -1)
    
    def _project_goal(self, robot_position: np.ndarray, robot_orientation: float,
                     goal: Goal, image: np.ndarray):
        """Project a goal onto the camera image."""
        # Transform goal position to camera coordinates
        world_pos = goal.position - robot_position
        
        # Rotate to camera frame
        cos_rot = np.cos(-robot_orientation)
        sin_rot = np.sin(-robot_orientation)
        camera_x = world_pos[0] * cos_rot - world_pos[1] * sin_rot
        camera_y = world_pos[0] * sin_rot + world_pos[1] * cos_rot
        
        # Skip if behind camera
        if camera_x <= 0:
            return
        
        # Project to image coordinates
        focal_length = self.resolution[0] / (2 * np.tan(self.field_of_view / 2))
        
        pixel_x = int(focal_length * camera_y / camera_x + self.resolution[0] / 2)
        pixel_y = int(focal_length * camera_x / camera_x + self.resolution[1] / 2)
        
        # Check if in image bounds
        if (0 <= pixel_x < self.resolution[0] and 
            0 <= pixel_y < self.resolution[1]):
            # Draw goal as a circle
            radius = int(goal.radius * focal_length / camera_x)
            cv2.circle(image, (pixel_x, pixel_y), radius, 0.3, 2)


class IMUSensor:
    """
    IMU (Inertial Measurement Unit) sensor simulation.
    
    Simulates an IMU providing position, velocity, and orientation
    measurements with realistic noise characteristics.
    """
    
    def __init__(self, noise_std: float = 0.01):
        """
        Initialize IMU sensor.
        
        Args:
            noise_std: Standard deviation of measurement noise
        """
        self.noise_std = noise_std
        self.prev_position = None
        self.prev_velocity = None
    
    def get_observation(self, robot_position: np.ndarray, robot_velocity: np.ndarray,
                       robot_orientation: float) -> np.ndarray:
        """
        Get IMU observation.
        
        Args:
            robot_position: Current robot position [x, y]
            robot_velocity: Current robot velocity [vx, vy]
            robot_orientation: Current robot orientation (radians)
            
        Returns:
            IMU measurements [pos_x, pos_y, vel_x, vel_y, orientation, angular_velocity]
        """
        # Calculate angular velocity (simplified)
        if self.prev_velocity is not None:
            # Estimate angular velocity from velocity change
            velocity_change = robot_velocity - self.prev_velocity
            angular_velocity = np.arctan2(velocity_change[1], velocity_change[0])
        else:
            angular_velocity = 0.0
        
        # Create measurement vector
        measurement = np.array([
            robot_position[0],    # pos_x
            robot_position[1],    # pos_y
            robot_velocity[0],    # vel_x
            robot_velocity[1],    # vel_y
            robot_orientation,    # orientation
            angular_velocity      # angular_velocity
        ], dtype=np.float32)
        
        # Add noise
        noise = np.random.normal(0, self.noise_std, 6)
        measurement += noise
        
        # Update previous values
        self.prev_position = robot_position.copy()
        self.prev_velocity = robot_velocity.copy()
        
        return measurement
    
    def get_readings(self, robot_state: dict) -> np.ndarray:
        """Alias for get_observation for backward compatibility."""
        return self.get_observation(
            robot_state['position'], 
            robot_state['velocity'], 
            robot_state['orientation']
        ) 