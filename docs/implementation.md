# Implementation Details

## Overview

This document provides detailed technical information about the implementation of the Adaptive Robot Navigation System using Active Inference. It covers the architecture, design decisions, and implementation specifics.

## System Architecture

### High-Level Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │    │Active Inference │    │   Evaluation    │
│                 │    │                 │    │                 │
│ • Navigation    │◄──►│ • Generative    │◄──►│ • Metrics       │
│ • Sensors       │    │ • Recognition   │    │ • Comparison    │
│ • Physics       │    │ • Policy        │    │ • Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agents        │    │ Visualization   │    │   Testing       │
│                 │    │                 │    │                 │
│ • Active Inf    │    │ • Navigation    │    │ • Unit Tests    │
│ • PPO           │    │ • Belief States │    │ • Integration   │
│ • DQN           │    │ • Free Energy   │    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Environment System (`src/environment/`)

**NavigationEnvironment** (`navigation_env.py`)
- 2D grid world with continuous robot movement
- Dynamic obstacle generation and movement
- Multi-sensor integration (LIDAR, camera, IMU)
- Physics simulation with collision detection
- Goal management and reward computation

**Sensor Models** (`sensors.py`)
- **LIDARSensor**: Ray-casting based distance measurements
- **CameraSensor**: 2D image generation with field of view
- **IMUSensor**: Position, velocity, and orientation tracking

#### 2. Active Inference Framework (`src/active_inference/`)

**GenerativeModel** (`generative_model.py`)
- Neural network-based observation model: p(o|s)
- Transition model: p(s'|s,a)
- Prior model: p(s)
- Probabilistic sampling and log-likelihood computation

**RecognitionModel** (`recognition_model.py`)
- Variational encoder: q(s|o)
- State inference and sampling
- KL divergence computation

**PolicyModel** (`policy_model.py`)
- Expected Free Energy computation
- Action selection via EFE minimization
- Information gain and utility balancing

**VariationalInference** (`variational_inference.py`)
- Free energy computation and optimization
- Belief update algorithms
- Model training and optimization

#### 3. Agent Implementations (`src/agents/`)

**ActiveInferenceAgent** (`active_inference_agent.py`)
- Integration of all active inference components
- Experience replay and online learning
- Exploration vs exploitation strategies
- Belief state tracking and uncertainty quantification

**RL Baseline Agents** (`rl_agents.py`)
- **PPOAgent**: Proximal Policy Optimization implementation
- **DQNAgent**: Deep Q-Network implementation
- Standard RL training and evaluation

#### 4. Evaluation Framework (`src/evaluation/`)

**NavigationMetrics** (`metrics.py`)
- Performance metrics calculation
- Statistical analysis tools
- Learning curve analysis
- Adaptability and efficiency measures

**AgentComparison** (`comparison.py`)
- Systematic agent comparison
- Multi-scenario evaluation
- Statistical significance testing
- Comprehensive reporting

#### 5. Visualization System (`src/visualization/`)

**NavigationVisualizer** (`navigation_visualizer.py`)
- Real-time environment visualization
- Robot trajectory and obstacle display
- Goal and reward visualization

**BeliefVisualizer** (`belief_visualizer.py`)
- Belief state visualization
- Uncertainty quantification display
- Free energy dynamics monitoring

## Implementation Details

### Neural Network Architecture

#### Generative Model Networks

```python
# Observation Model: p(o|s)
observation_model = nn.Sequential(
    nn.Linear(state_dim, hidden_dims[0]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[0], hidden_dims[1]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[1], observation_dim * 2)  # mean + logvar
)

# Transition Model: p(s'|s,a)
transition_model = nn.Sequential(
    nn.Linear(state_dim + action_dim, hidden_dims[0]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[0], hidden_dims[1]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[1], state_dim * 2)  # mean + logvar
)
```

#### Recognition Model Network

```python
# Recognition Model: q(s|o)
encoder = nn.Sequential(
    nn.Linear(observation_dim, hidden_dims[0]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[0], hidden_dims[1]),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dims[1], state_dim * 2)  # mean + logvar
)
```

### Active Inference Algorithm

#### 1. Belief Update

```python
def update_beliefs(self, observation, num_iterations=10):
    """Update beliefs through variational inference."""
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
    
    for _ in range(num_iterations):
        # Forward pass through recognition model
        belief_outputs = self.recognition_model(obs_tensor)
        
        # Sample states
        states = self.recognition_model.sample_states(obs_tensor)
        
        # Compute free energy
        fe = self.variational_inference.compute_free_energy(
            obs_tensor, states, None, None,
            self.generative_model, self.recognition_model
        )
        
        # Update recognition model
        self.recognition_optimizer.zero_grad()
        fe.mean().backward()
        self.recognition_optimizer.step()
    
    return belief_outputs
```

#### 2. Action Selection

```python
def select_action_by_efe(self, states):
    """Select action by minimizing Expected Free Energy."""
    # Compute EFE for all actions
    efe_matrix = self.policy_model.compute_policy_efe(
        states, self.generative_model, self.recognition_model
    )
    
    # Select action with minimum EFE
    action = torch.argmin(efe_matrix, dim=1).item()
    return action
```

#### 3. Expected Free Energy Computation

```python
def compute_expected_free_energy(self, states, generative_model, recognition_model):
    """Compute Expected Free Energy for action selection."""
    batch_size = states.shape[0]
    efe_matrix = torch.zeros(batch_size, self.action_dim)
    
    for action in range(self.action_dim):
        # Sample next states
        actions = torch.full((batch_size,), action, dtype=torch.long)
        next_states = generative_model.sample_next_states(states, actions)
        
        # Sample observations
        observations = generative_model.sample_observations(next_states)
        
        # Compute information gain (KL divergence)
        info_gain = self.compute_information_gain(states, next_states)
        
        # Compute utility (negative distance to goal)
        utility = self.compute_utility(next_states)
        
        # Combine terms
        efe_matrix[:, action] = (
            self.efe_weight * (info_gain + utility)
        )
    
    return efe_matrix
```

### Environment Implementation

#### Sensor Integration

```python
def _get_observation(self):
    """Generate complete observation from all sensors."""
    # LIDAR readings
    lidar_readings = self.lidar.get_readings(
        self.robot.position, self.robot.orientation,
        [obs.position for obs in self.static_obstacles + self.dynamic_obstacles]
    )
    
    # Camera image
    camera_image = self.camera.get_image(
        self.robot.position, self.robot.orientation,
        [obs.position for obs in self.static_obstacles + self.dynamic_obstacles]
    )
    
    # IMU readings
    robot_state = {
        'position': self.robot.position,
        'velocity': self.robot.velocity,
        'orientation': self.robot.orientation
    }
    imu_readings = self.imu.get_readings(robot_state)
    
    # Goal information
    goal_info = self._get_goal_observation()
    
    # Concatenate all observations
    observation = np.concatenate([
        lidar_readings,
        camera_image.flatten(),
        imu_readings,
        goal_info
    ])
    
    return observation
```

#### Physics Simulation

```python
def _apply_action(self, action):
    """Apply action to robot and update physics."""
    # Convert action to movement
    if action == 0:  # Forward
        movement = np.array([0, self.max_speed])
    elif action == 1:  # Backward
        movement = np.array([0, -self.max_speed])
    elif action == 2:  # Left
        movement = np.array([-self.max_speed, 0])
    else:  # Right
        movement = np.array([self.max_speed, 0])
    
    # Update robot position
    new_position = self.robot.position + movement
    
    # Check boundaries
    new_position[0] = np.clip(new_position[0], 0, self.width)
    new_position[1] = np.clip(new_position[1], 0, self.height)
    
    # Update robot state
    self.robot.position = new_position
    self.robot.velocity = movement
    
    # Drain battery
    self.robot.battery -= 0.1
```

### Training and Optimization

#### Experience Replay

```python
def store_experience(self, observation, action, reward, next_observation, done):
    """Store experience in replay buffer."""
    self.memory.append({
        'observation': observation,
        'action': action,
        'reward': reward,
        'next_observation': next_observation,
        'done': done
    })
```

#### Model Updates

```python
def update_models(self):
    """Update all models using stored experiences."""
    if len(self.memory) < self.batch_size:
        return {}
    
    # Sample batch
    batch = random.sample(self.memory, self.batch_size)
    
    # Prepare data
    observations = torch.FloatTensor([exp['observation'] for exp in batch])
    actions = torch.LongTensor([exp['action'] for exp in batch])
    rewards = torch.FloatTensor([exp['reward'] for exp in batch])
    next_observations = torch.FloatTensor([exp['next_observation'] for exp in batch])
    
    # Update models
    losses = self.variational_inference.optimize_models(
        observations, None, actions, rewards,
        self.generative_model, self.recognition_model, self.policy_model
    )
    
    return losses
```

## Configuration System

### Environment Configuration

```yaml
environment:
  width: 50
  height: 50
  robot:
    initial_position: [25, 25]
    max_speed: 1.0
    sensor_range: 10
  obstacles:
    num_static: 15
    num_dynamic: 5
    dynamic_speed: 0.5
  sensors:
    lidar:
      num_rays: 36
      max_range: 10.0
      noise_std: 0.1
    camera:
      resolution: [64, 64]
      field_of_view: 90
      noise_std: 0.05
    imu:
      noise_std: 0.01
```

### Agent Configuration

```yaml
agent:
  active_inference:
    state_dim: 64
    observation_dim: 128
    action_dim: 4
    precision: 1.0
    learning_rate: 0.01
    temperature: 1.0
    num_samples: 10
    kl_weight: 1.0
  policy:
    exploration_rate: 0.1
    exploration_decay: 0.995
    min_exploration_rate: 0.01
    efe_weight: 1.0
    information_gain_weight: 0.5
    utility_weight: 0.5
  neural_networks:
    generative:
      hidden_dims: [128, 64, 32]
      activation: "relu"
      dropout: 0.1
    recognition:
      hidden_dims: [128, 64, 32]
      activation: "relu"
      dropout: 0.1
    policy:
      hidden_dims: [64, 32]
      activation: "relu"
  training:
    batch_size: 32
    update_frequency: 10
    memory_size: 10000
```

## Usage Examples

### Basic Navigation

```python
from src.environment import NavigationEnvironment
from src.agents import ActiveInferenceAgent

# Create environment and agent
env = NavigationEnvironment()
agent = ActiveInferenceAgent(env.observation_space, env.action_space)

# Run navigation
obs = env.reset()
for step in range(1000):
    action = agent.select_action(obs, training=True)
    obs, reward, done, info = env.step(action)
    agent.step(obs, reward, done, info)
    
    if done:
        break
```

### Comparison Experiment

```python
from src.evaluation import AgentComparison

# Create comparison framework
comparison = AgentComparison()

# Run comparison
results = comparison.run_quick_comparison(num_episodes=30)

# Generate report
comparison.metrics.generate_report(results)
```

### Visualization

```python
from src.visualization import NavigationVisualizer, BeliefVisualizer

# Create visualizers
nav_viz = NavigationVisualizer()
belief_viz = BeliefVisualizer()

# Update and render
nav_viz.update(env, agent)
nav_viz.render()

belief_viz.update(agent.current_beliefs)
belief_viz.render()
```

## Performance Considerations

### Computational Efficiency

1. **Batch Processing**: All neural network operations use batch processing for efficiency
2. **GPU Acceleration**: Models automatically use CUDA if available
3. **Memory Management**: Experience replay uses deque with maximum size
4. **Optimization**: Gradient clipping and adaptive learning rates

### Memory Usage

- **Observation Space**: ~128 dimensions (configurable)
- **State Space**: ~64 dimensions (configurable)
- **Replay Buffer**: 10,000 experiences (configurable)
- **Model Parameters**: ~100K-500K parameters depending on architecture

### Scalability

- **Environment Size**: Configurable grid dimensions
- **Number of Obstacles**: Configurable static and dynamic obstacles
- **Sensor Resolution**: Configurable LIDAR rays and camera resolution
- **Agent Complexity**: Configurable neural network architectures

## Testing and Validation

### Unit Tests

- Environment functionality and physics
- Sensor models and noise
- Active inference components
- Agent implementations
- Integration tests

### Performance Tests

- Training convergence
- Memory usage
- Computational efficiency
- Agent comparison benchmarks

### Validation Metrics

- Success rate and efficiency
- Learning curves
- Statistical significance
- Adaptability measures

## Future Extensions

### Planned Enhancements

1. **3D Environment**: Extend to 3D navigation
2. **Multi-Agent**: Support for multiple robots
3. **Real Sensors**: Integration with real sensor hardware
4. **Advanced AI**: Hierarchical active inference
5. **Distributed**: Multi-GPU and distributed training

### Research Directions

1. **Meta-Learning**: Fast adaptation to new environments
2. **Hierarchical Control**: Multi-level decision making
3. **Social Navigation**: Human-aware navigation
4. **Energy Efficiency**: Optimized power consumption
5. **Safety**: Formal safety guarantees 