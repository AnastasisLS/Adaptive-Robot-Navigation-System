# API Documentation

## Overview

This document provides comprehensive API documentation for the Adaptive Robot Navigation System using Active Inference. It covers all major classes, methods, and their usage.

## Environment System

### NavigationEnvironment

The main environment class for robot navigation simulation.

#### Constructor

```python
NavigationEnvironment(config_path: str = "config/environment_config.yaml")
```

**Parameters:**
- `config_path` (str): Path to environment configuration file

**Returns:**
- `NavigationEnvironment`: Initialized environment instance

#### Methods

##### reset()
Reset the environment to initial state.

```python
observation = env.reset()
```

**Returns:**
- `np.ndarray`: Initial observation

##### step(action)
Take a step in the environment.

```python
next_observation, reward, done, info = env.step(action)
```

**Parameters:**
- `action` (int): Action to take (0-3: forward, backward, left, right)

**Returns:**
- `next_observation` (np.ndarray): New observation
- `reward` (float): Reward received
- `done` (bool): Whether episode is done
- `info` (dict): Additional information

##### render(mode='human')
Render the environment.

```python
image = env.render(mode='human')
```

**Parameters:**
- `mode` (str): Rendering mode ('human' or 'rgb_array')

**Returns:**
- `np.ndarray` or `None`: Rendered image

#### Properties

- `observation_space` (int): Dimension of observation space
- `action_space` (int): Number of possible actions
- `width` (int): Environment width
- `height` (int): Environment height
- `robot` (RobotState): Current robot state
- `goals` (List[Goal]): List of goals
- `static_obstacles` (List[Obstacle]): List of static obstacles
- `dynamic_obstacles` (List[Obstacle]): List of dynamic obstacles

### Sensor Classes

#### LIDARSensor

```python
class LIDARSensor:
    def __init__(self, num_rays: int, max_range: float, noise_std: float)
    
    def get_readings(self, robot_pos: np.ndarray, robot_orientation: float, 
                    obstacles: List[np.ndarray]) -> np.ndarray
```

#### CameraSensor

```python
class CameraSensor:
    def __init__(self, resolution: Tuple[int, int], field_of_view: float, noise_std: float)
    
    def get_image(self, robot_pos: np.ndarray, robot_orientation: float,
                  obstacles: List[np.ndarray]) -> np.ndarray
```

#### IMUSensor

```python
class IMUSensor:
    def __init__(self, noise_std: float)
    
    def get_readings(self, robot_state: Dict[str, Any]) -> np.ndarray
```

## Active Inference Framework

### GenerativeModel

Neural network-based generative model for active inference.

#### Constructor

```python
GenerativeModel(config_path: str = "config/agent_config.yaml")
```

#### Methods

##### forward(states, actions=None)
Forward pass through the generative model.

```python
outputs = model.forward(states, actions)
```

**Parameters:**
- `states` (torch.Tensor): Hidden states [batch_size, state_dim]
- `actions` (torch.Tensor, optional): Actions [batch_size, action_dim]

**Returns:**
- `dict`: Dictionary containing model outputs

##### sample_observations(states)
Sample observations from the observation model.

```python
observations = model.sample_observations(states)
```

**Returns:**
- `torch.Tensor`: Sampled observations [batch_size, observation_dim]

##### sample_next_states(states, actions)
Sample next states from the transition model.

```python
next_states = model.sample_next_states(states, actions)
```

**Returns:**
- `torch.Tensor`: Sampled next states [batch_size, state_dim]

### RecognitionModel

Variational encoder for belief state inference.

#### Constructor

```python
RecognitionModel(config_path: str = "config/agent_config.yaml")
```

#### Methods

##### forward(observations)
Forward pass through the recognition model.

```python
outputs = model.forward(observations)
```

**Returns:**
- `dict`: Dictionary containing state mean and log variance

##### sample_states(observations)
Sample states from the recognition model.

```python
states = model.sample_states(observations)
```

**Returns:**
- `torch.Tensor`: Sampled states [batch_size, state_dim]

##### compute_kl_divergence(mean1, logvar1, mean2, logvar2)
Compute KL divergence between two Gaussian distributions.

```python
kl_div = model.compute_kl_divergence(mean1, logvar1, mean2, logvar2)
```

**Returns:**
- `torch.Tensor`: KL divergence values [batch_size]

### PolicyModel

Policy model for action selection via Expected Free Energy.

#### Constructor

```python
PolicyModel(config_path: str = "config/agent_config.yaml")
```

#### Methods

##### compute_policy_efe(states, generative_model, recognition_model)
Compute Expected Free Energy for all actions.

```python
efe_matrix = model.compute_policy_efe(states, generative_model, recognition_model)
```

**Returns:**
- `torch.Tensor`: EFE matrix [batch_size, action_dim]

##### select_action(states, generative_model, recognition_model)
Select action by minimizing EFE.

```python
action = model.select_action(states, generative_model, recognition_model)
```

**Returns:**
- `torch.Tensor`: Selected actions [batch_size]

### VariationalInference

Variational inference engine for belief updates and model optimization.

#### Constructor

```python
VariationalInference(config_path: str = "config/agent_config.yaml")
```

#### Methods

##### compute_free_energy(observations, states, actions, rewards, generative_model, recognition_model)
Compute variational free energy.

```python
free_energy = model.compute_free_energy(observations, states, actions, rewards, 
                                       generative_model, recognition_model)
```

**Returns:**
- `torch.Tensor`: Free energy values [batch_size]

##### optimize_models(observations, states, actions, rewards, generative_model, recognition_model, policy_model)
Optimize all models using variational inference.

```python
losses = model.optimize_models(observations, states, actions, rewards,
                              generative_model, recognition_model, policy_model)
```

**Returns:**
- `dict`: Dictionary containing training losses

##### update_beliefs(observations, recognition_model, num_iterations=10)
Update beliefs through iterative variational inference.

```python
updated_states = model.update_beliefs(observations, recognition_model, num_iterations)
```

**Returns:**
- `torch.Tensor`: Updated belief states [batch_size, state_dim]

## Agent Implementations

### ActiveInferenceAgent

Complete active inference agent implementation.

#### Constructor

```python
ActiveInferenceAgent(observation_space: int, action_space: int, 
                    config_path: str = "config/agent_config.yaml")
```

#### Methods

##### select_action(observation, training=True)
Select action based on current observation.

```python
action = agent.select_action(observation, training=True)
```

**Parameters:**
- `observation` (np.ndarray): Current observation
- `training` (bool): Whether in training mode

**Returns:**
- `int`: Selected action

##### step(observation, reward, done, info)
Process step information and update agent.

```python
agent.step(observation, reward, done, info)
```

##### store_experience(observation, action, reward, next_observation, done)
Store experience in replay buffer.

```python
agent.store_experience(observation, action, reward, next_observation, done)
```

##### update_models()
Update all models using stored experiences.

```python
losses = agent.update_models()
```

**Returns:**
- `dict`: Training losses

##### update_beliefs(observation, num_iterations=10)
Update belief states for given observation.

```python
beliefs = agent.update_beliefs(observation, num_iterations)
```

**Returns:**
- `dict`: Updated belief states

##### get_belief_uncertainty(observation)
Get belief uncertainty for given observation.

```python
uncertainty = agent.get_belief_uncertainty(observation)
```

**Returns:**
- `np.ndarray`: Uncertainty values

##### get_action_probabilities(observation)
Get action probabilities for given observation.

```python
probabilities = agent.get_action_probabilities(observation)
```

**Returns:**
- `np.ndarray`: Action probabilities

##### get_free_energy(observation)
Get free energy for given observation.

```python
free_energy = agent.get_free_energy(observation)
```

**Returns:**
- `float`: Free energy value

##### save_models(path)
Save all models to file.

```python
agent.save_models("models/agent.pth")
```

##### load_models(path)
Load all models from file.

```python
agent.load_models("models/agent.pth")
```

##### get_training_stats()
Get training statistics.

```python
stats = agent.get_training_stats()
```

**Returns:**
- `dict`: Training statistics

### RL Baseline Agents

#### PPOAgent

```python
class PPOAgent:
    def __init__(self, observation_space: int, action_space: int, 
                 config_path: str = "config/agent_config.yaml")
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int
    def save_model(self, path: str)
    def load_model(self, path: str)
```

#### DQNAgent

```python
class DQNAgent:
    def __init__(self, observation_space: int, action_space: int, 
                 config_path: str = "config/agent_config.yaml")
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int
    def save_model(self, path: str)
    def load_model(self, path: str)
```

## Evaluation Framework

### NavigationMetrics

Metrics calculation and analysis for navigation performance.

#### Constructor

```python
NavigationMetrics()
```

#### Methods

##### calculate_episode_metrics(episode_data)
Calculate metrics for a single episode.

```python
metrics = calculator.calculate_episode_metrics(episode_data)
```

**Parameters:**
- `episode_data` (dict): Episode data dictionary

**Returns:**
- `dict`: Calculated metrics

##### calculate_aggregate_metrics(episodes_data)
Calculate aggregate metrics across multiple episodes.

```python
aggregate = calculator.calculate_aggregate_metrics(episodes_data)
```

**Returns:**
- `dict`: Aggregate metrics

##### compare_agents(agent_results)
Compare performance between different agents.

```python
comparison = calculator.compare_agents(agent_results)
```

**Returns:**
- `dict`: Comparison results

##### generate_report(agent_results, save_path=None)
Generate comprehensive performance report.

```python
report = calculator.generate_report(agent_results, save_path="report.txt")
```

**Returns:**
- `str`: Report text

##### plot_comparison(agent_results, save_path=None)
Create comparison plots for different agents.

```python
calculator.plot_comparison(agent_results, save_path="comparison.png")
```

### AgentComparison

Comprehensive comparison framework for different agent types.

#### Constructor

```python
AgentComparison(config_path: str = "config/agent_config.yaml")
```

#### Methods

##### run_comparison_experiment(agents, num_episodes=50, max_steps=1000, scenarios=None)
Run comprehensive comparison experiment.

```python
results = comparison.run_comparison_experiment(agents, num_episodes=50)
```

**Returns:**
- `dict`: Comparison results

##### create_agent_ensemble()
Create a standard set of agents for comparison.

```python
agents = comparison.create_agent_ensemble()
```

**Returns:**
- `dict`: Dictionary of agents

##### run_quick_comparison(num_episodes=20)
Run a quick comparison with default agents.

```python
results = comparison.run_quick_comparison(num_episodes=30)
```

**Returns:**
- `dict`: Quick comparison results

## Visualization System

### NavigationVisualizer

Real-time navigation environment visualization.

#### Constructor

```python
NavigationVisualizer()
```

#### Methods

##### update(environment, agent)
Update visualization with current environment and agent state.

```python
visualizer.update(env, agent)
```

##### render()
Render the visualization.

```python
visualizer.render()
```

### BeliefVisualizer

Belief state and uncertainty visualization.

#### Constructor

```python
BeliefVisualizer()
```

#### Methods

##### update(beliefs)
Update visualization with current belief states.

```python
visualizer.update(agent.current_beliefs)
```

##### render()
Render the belief visualization.

```python
visualizer.render()
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
  physics:
    collision_penalty: -10.0
    step_penalty: -0.1
    max_steps: 1000
  dynamics:
    obstacle_movement: true
    goal_changes: true
    change_frequency: 100
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
  optimization:
    optimizer: "adam"
    learning_rate: 0.001
    weight_decay: 1e-5
    gradient_clip: 1.0
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

### Custom Configuration

```python
import yaml

# Load custom configuration
with open("custom_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create environment with custom config
env = NavigationEnvironment("custom_config.yaml")

# Create agent with custom config
agent = ActiveInferenceAgent(
    env.observation_space, 
    env.action_space,
    config_path="custom_agent_config.yaml"
)
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Configuration file not found
- `ValueError`: Invalid parameter values
- `RuntimeError`: Model loading/saving errors
- `ImportError`: Missing dependencies

### Debugging Tips

1. Check configuration file paths and syntax
2. Verify observation and action space dimensions
3. Monitor free energy and belief uncertainty
4. Use visualization tools for debugging
5. Check training statistics and losses

## Performance Considerations

### Memory Usage

- Observation space: ~128 dimensions
- State space: ~64 dimensions
- Replay buffer: 10,000 experiences
- Model parameters: ~100K-500K

### Computational Requirements

- GPU recommended for training
- CPU sufficient for inference
- Batch processing for efficiency
- Gradient clipping for stability

### Optimization Tips

1. Use appropriate batch sizes
2. Monitor free energy convergence
3. Adjust learning rates based on performance
4. Regularize neural networks with dropout
5. Use experience replay for stability 