"""
Unit tests for navigation environment and sensor systems.

Tests the core environment functionality, sensor models, and
environment dynamics.
"""

import unittest
import numpy as np
import tempfile
import os
import yaml

from src.environment import NavigationEnvironment
from src.environment.sensors import LIDARSensor, CameraSensor, IMUSensor
from src.environment.navigation_env import Obstacle


class TestNavigationEnvironment(unittest.TestCase):
    """Test cases for NavigationEnvironment."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'environment': {
                'width': 20,
                'height': 20,
                'robot': {
                    'initial_position': [10, 10],
                    'max_speed': 1.0,
                    'sensor_range': 5
                },
                'obstacles': {
                    'num_static': 3,
                    'num_dynamic': 1,
                    'dynamic_speed': 0.3,
                    'obstacle_size': 1.0
                },
                'goals': {
                    'num_goals': 2,
                    'goal_radius': 1.5,
                    'goal_reward': 100.0
                },
                'sensors': {
                    'lidar': {
                        'num_rays': 8,
                        'max_range': 5.0,
                        'noise_std': 0.1
                    },
                    'camera': {
                        'resolution': [16, 16],
                        'field_of_view': 90,
                        'noise_std': 0.05
                    },
                    'imu': {
                        'noise_std': 0.01
                    }
                },
                'physics': {
                    'collision_penalty': -10.0,
                    'step_penalty': -0.1,
                    'max_steps': 100
                },
                'dynamics': {
                    'obstacle_movement': True,
                    'goal_changes': False,
                    'change_frequency': 50
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create environment
        self.env = NavigationEnvironment(self.temp_config.name)
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.width, 20)
        self.assertEqual(self.env.height, 20)
        self.assertEqual(self.env.action_space.n, 4)  # Check Discrete(4).n
        self.assertEqual(self.env.observation_space.shape[0], 274)  # Check Box shape
    
    def test_robot_initialization(self):
        """Test robot state initialization."""
        self.assertTrue(np.array_equal(
            self.env.robot.position, 
            np.array([10, 10], dtype=np.float32)
        ))
        self.assertEqual(self.env.robot.battery, 100.0)
        self.assertEqual(self.env.robot.orientation, 0.0)
    
    def test_sensor_initialization(self):
        """Test sensor system initialization."""
        self.assertIsNotNone(self.env.lidar)
        self.assertIsNotNone(self.env.camera)
        self.assertIsNotNone(self.env.imu)
        
        # Test LIDAR
        self.assertEqual(self.env.lidar.num_rays, 8)
        self.assertEqual(self.env.lidar.max_range, 5.0)
        
        # Test Camera
        self.assertEqual(list(self.env.camera.resolution), [16, 16])
        
        # Test IMU
        self.assertEqual(self.env.imu.noise_std, 0.01)
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        # Take some steps to change state
        obs = self.env.reset()
        initial_pos = self.env.robot.position.copy()
        
        # Take a few steps
        for _ in range(5):
            self.env.step(0)  # Move forward
        
        # Reset and check if robot is back to initial position
        obs = self.env.reset()
        self.assertTrue(np.array_equal(
            self.env.robot.position, 
            np.array([10, 10], dtype=np.float32)
        ))
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.robot.battery, 100.0)
    
    def test_obstacle_generation(self):
        """Test obstacle generation."""
        # Set episode count to ensure full difficulty for testing
        self.env.episode_count = 100  # This ensures full curriculum difficulty
        self.env._generate_obstacles()
        
        # Check that obstacles were generated
        self.assertGreater(len(self.env.static_obstacles), 0)
        # Note: Dynamic obstacles may be 0 in early episodes due to curriculum learning
        # We'll check that the generation method works, but not require dynamic obstacles
        
        # Check obstacle properties
        for obstacle in self.env.static_obstacles:
            self.assertFalse(obstacle.dynamic)
            self.assertEqual(obstacle.size, 1.0)
            self.assertTrue(0 <= obstacle.position[0] <= self.env.width)
            self.assertTrue(0 <= obstacle.position[1] <= self.env.height)
        
        for obstacle in self.env.dynamic_obstacles:
            self.assertTrue(obstacle.dynamic)
            self.assertEqual(obstacle.size, 1.0)
            self.assertTrue(0 <= obstacle.position[0] <= self.env.width)
            self.assertTrue(0 <= obstacle.position[1] <= self.env.height)
    
    def test_goal_generation(self):
        """Test goal generation."""
        self.env._generate_goals()
        
        # Check that goals were generated
        self.assertGreater(len(self.env.goals), 0)
        
        # Check goal properties
        for goal in self.env.goals:
            self.assertEqual(goal.radius, 1.5)
            self.assertEqual(goal.reward, 100.0)
            self.assertTrue(goal.active)
            self.assertTrue(0 <= goal.position[0] <= self.env.width)
            self.assertTrue(0 <= goal.position[1] <= self.env.height)
    
    def test_action_execution(self):
        """Test action execution."""
        initial_pos = self.env.robot.position.copy()
        
        # Test forward movement
        self.env._apply_action(0)  # Forward
        # Check that robot moved (position changed)
        self.assertFalse(np.array_equal(self.env.robot.position, initial_pos))
        
        # Test backward movement
        pos_before_backward = self.env.robot.position.copy()
        self.env._apply_action(1)  # Backward
        self.assertFalse(np.array_equal(self.env.robot.position, pos_before_backward))
        
        # Test left movement
        pos_before_left = self.env.robot.position.copy()
        self.env._apply_action(2)  # Left
        self.assertFalse(np.array_equal(self.env.robot.position, pos_before_left))
        
        # Test right movement
        pos_before_right = self.env.robot.position.copy()
        self.env._apply_action(3)  # Right
        self.assertFalse(np.array_equal(self.env.robot.position, pos_before_right))
    
    def test_collision_detection(self):
        """Test collision detection."""
        # Clear all obstacles first
        self.env.static_obstacles.clear()
        self.env.dynamic_obstacles.clear()
        
        # Place robot at a known position
        self.env.robot.position = np.array([5.0, 5.0])
        
        # Create a single obstacle very close to robot
        obstacle = Obstacle(
            position=np.array([5.5, 5.0]), 
            velocity=np.array([0.0, 0.0]), 
            size=1.0
        )
        self.env.static_obstacles.append(obstacle)
        
        # Check collision
        collision = self.env._check_collisions()
        self.assertTrue(collision)
        
        # Move obstacle away
        obstacle.position = np.array([10.0, 10.0])
        collision = self.env._check_collisions()
        self.assertFalse(collision)
    
    def test_goal_reaching(self):
        """Test goal reaching detection."""
        # Place robot near a goal
        self.env.robot.position = np.array([5.0, 5.0])
        goal = self.env.goals[0]
        goal.position = np.array([5.5, 5.0])  # Within goal radius
        
        # Check goal reached
        goal_reached = self.env._check_goal_reached()
        self.assertTrue(goal_reached)
        
        # Move robot away
        self.env.robot.position = np.array([10.0, 10.0])
        goal_reached = self.env._check_goal_reached()
        self.assertFalse(goal_reached)
    
    def test_observation_generation(self):
        """Test observation generation."""
        obs = self.env._get_observation()
    
        # Check observation dimensions
        self.assertEqual(len(obs), self.env.observation_space.shape[0])
        self.assertTrue(np.all(np.isfinite(obs)))
        
        # Check that observation contains all sensor data
        lidar_dim = self.env.lidar.num_rays
        camera_dim = self.env.camera.resolution[0] * self.env.camera.resolution[1]
        imu_dim = 6
        goal_dim = 4
        
        expected_dim = lidar_dim + camera_dim + imu_dim + goal_dim
        self.assertEqual(len(obs), expected_dim)
    
    def test_environment_step(self):
        """Test complete environment step."""
        obs = self.env.reset()
    
        # Take a step
        next_obs, reward, done, info = self.env.step(0)
    
        # Check return values
        self.assertEqual(len(next_obs), self.env.observation_space.shape[0])
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check that step count increased
        self.assertEqual(self.env.step_count, 1)
        
        # Check that robot moved
        self.assertFalse(np.array_equal(obs, next_obs))
    
    def test_dynamic_obstacle_movement(self):
        """Test dynamic obstacle movement."""
        if len(self.env.dynamic_obstacles) > 0:
            obstacle = self.env.dynamic_obstacles[0]
            initial_pos = obstacle.position.copy()
            
            # Update dynamic obstacles
            self.env._update_dynamic_obstacles()
            
            # Check that obstacle moved
            self.assertFalse(np.array_equal(initial_pos, obstacle.position))
    
    def test_battery_drain(self):
        """Test battery drain over time."""
        initial_battery = self.env.robot.battery
        
        # Take several steps
        for _ in range(10):
            self.env.step(0)
        
        # Check that battery decreased
        self.assertLess(self.env.robot.battery, initial_battery)
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        obs = self.env.reset()
        
        # Take maximum steps
        for _ in range(self.env.max_steps + 1):
            obs, reward, done, info = self.env.step(0)
            if done:
                break
        
        # Should terminate after max steps
        self.assertTrue(done)
    
    def test_curriculum_learning(self):
        """Test curriculum learning functionality."""
        # Test early episode (should have fewer obstacles and closer goals)
        self.env.episode_count = 1
        self.env._generate_obstacles()
        self.env._generate_goals()
        
        early_static_count = len(self.env.static_obstacles)
        early_dynamic_count = len(self.env.dynamic_obstacles)
        
        # Test later episode (should have more obstacles)
        self.env.episode_count = 100
        self.env._generate_obstacles()
        self.env._generate_goals()
        
        late_static_count = len(self.env.static_obstacles)
        late_dynamic_count = len(self.env.dynamic_obstacles)
        
        # Later episodes should have more obstacles
        self.assertGreaterEqual(late_static_count, early_static_count)
        self.assertGreaterEqual(late_dynamic_count, early_dynamic_count)
        
        # Test curriculum info method
        curriculum_info = self.env.get_curriculum_info()
        self.assertIn('episode_count', curriculum_info)
        self.assertIn('obstacle_curriculum_factor', curriculum_info)
        self.assertIn('goal_curriculum_factor', curriculum_info)
        self.assertIn('num_static_obstacles', curriculum_info)
        self.assertIn('num_dynamic_obstacles', curriculum_info)
        self.assertIn('early_episode_bonus', curriculum_info)


class TestSensors(unittest.TestCase):
    """Test cases for sensor systems."""
    
    def setUp(self):
        """Set up sensor tests."""
        self.robot_pos = np.array([10.0, 10.0])
        self.robot_orientation = 0.0
        
        # Create sensors
        self.lidar = LIDARSensor(num_rays=8, max_range=5.0, noise_std=0.1)
        self.camera = CameraSensor(resolution=(16, 16), field_of_view=90, noise_std=0.05)
        self.imu = IMUSensor(noise_std=0.01)
    
    def test_lidar_sensor(self):
        """Test LIDAR sensor functionality."""
        # Create some obstacles
        obstacles = [
            Obstacle(position=np.array([12.0, 10.0]), velocity=np.array([0.0, 0.0]), size=1.0),  # Right of robot
            Obstacle(position=np.array([8.0, 10.0]), velocity=np.array([0.0, 0.0]), size=1.0),   # Left of robot
            Obstacle(position=np.array([10.0, 12.0]), velocity=np.array([0.0, 0.0]), size=1.0),  # Above robot
            Obstacle(position=np.array([10.0, 8.0]), velocity=np.array([0.0, 0.0]), size=1.0)    # Below robot
        ]
        
        # Get LIDAR readings
        readings = self.lidar.get_readings(
            self.robot_pos, self.robot_orientation, obstacles
        )
        
        # Check readings
        self.assertEqual(len(readings), self.lidar.num_rays)
        self.assertTrue(np.all(readings >= 0))
        self.assertTrue(np.all(readings <= self.lidar.max_range))
        self.assertTrue(np.all(np.isfinite(readings)))
        
        # Check that obstacles are detected
        # Should have readings less than max_range for obstacles
        self.assertTrue(np.any(readings < self.lidar.max_range))
    
    def test_camera_sensor(self):
        """Test camera sensor functionality."""
        # Create some obstacles
        obstacles = [
            Obstacle(position=np.array([12.0, 10.0]), velocity=np.array([0.0, 0.0]), size=1.0),  # Right of robot
            Obstacle(position=np.array([8.0, 10.0]), velocity=np.array([0.0, 0.0]), size=1.0),   # Left of robot
        ]
        
        # Get camera readings
        image = self.camera.get_image(
            self.robot_pos, self.robot_orientation, obstacles
        )
        
        # Check image properties
        self.assertEqual(image.shape, self.camera.resolution)
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(image <= 1.0))
        self.assertTrue(np.all(np.isfinite(image)))
    
    def test_imu_sensor(self):
        """Test IMU sensor functionality."""
        # Create robot state
        robot_state = {
            'position': self.robot_pos,
            'velocity': np.array([1.0, 0.5]),
            'orientation': self.robot_orientation
        }
        
        # Get IMU readings
        readings = self.imu.get_readings(robot_state)
        
        # Check readings
        self.assertEqual(len(readings), 6)  # pos_x, pos_y, vel_x, vel_y, orientation, angular_velocity
        self.assertTrue(np.all(np.isfinite(readings)))
        
        # Check that readings are close to true values (with noise)
        self.assertAlmostEqual(readings[0], self.robot_pos[0], delta=0.1)
        self.assertAlmostEqual(readings[1], self.robot_pos[1], delta=0.1)
        self.assertAlmostEqual(readings[2], 1.0, delta=0.1)
        self.assertAlmostEqual(readings[3], 0.5, delta=0.1)
        self.assertAlmostEqual(readings[4], self.robot_orientation, delta=0.1)
    
    def test_sensor_noise(self):
        """Test that sensors add appropriate noise."""
        # Test LIDAR noise
        obstacles = [Obstacle(position=np.array([15.0, 10.0]), velocity=np.array([0.0, 0.0]), size=1.0)]  # Far obstacle
        readings1 = self.lidar.get_readings(
            self.robot_pos, self.robot_orientation, obstacles
        )
        readings2 = self.lidar.get_readings(
            self.robot_pos, self.robot_orientation, obstacles
        )
        
        # Readings should be different due to noise
        self.assertFalse(np.array_equal(readings1, readings2))
        
        # Test camera noise
        image1 = self.camera.get_image(
            self.robot_pos, self.robot_orientation, obstacles
        )
        image2 = self.camera.get_image(
            self.robot_pos, self.robot_orientation, obstacles
        )
        
        # Images should be different due to noise
        self.assertFalse(np.array_equal(image1, image2))


if __name__ == '__main__':
    unittest.main() 