# Adaptive Robot Navigation System Using Active Inference

## Overview
Implementation of an autonomous robotic agent using active inference framework for adaptive navigation in dynamic environments.

## Key Components
- **Active Inference Agent**: Probabilistic world model with variational free energy minimization
- **Simulation Environment**: Dynamic navigation with obstacles and changing goals
- **Evaluation Framework**: Comparison with RL methods (PPO, DQN)
- **Visualization Tools**: Real-time belief state and navigation visualization

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start
```python
from src.environment import NavigationEnvironment
from src.agents import ActiveInferenceAgent

env = NavigationEnvironment()
agent = ActiveInferenceAgent(env.observation_space, env.action_space)

obs = env.reset()
for _ in range(1000):
    action = agent.select_action(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
```

## Project Structure
- `src/`: Core implementation (environment, agents, active inference)
- `examples/`: Usage examples and experiments
- `tests/`: Unit tests
- `docs/`: Documentation and theory
- `data/`: Experiment results and models

## Project Overview

This project implements a simulated autonomous robotic agent using the active inference framework. The agent navigates and adapts to dynamic environments by maintaining and updating probabilistic internal world models, continuously minimizing variational free energy ("surprise"). The system demonstrates adaptive navigation strategies and evaluates their performance against standard reinforcement learning methods.

## Theoretical Background

### Active Inference Framework

Active inference is a unified theory of brain function that treats perception, action, and learning as different aspects of the same underlying process: minimizing variational free energy. The key components include:

- **Generative Model**: A probabilistic model of how observations are generated from hidden states
- **Recognition Model**: An approximate posterior over hidden states given observations
- **Policy Selection**: Actions are chosen to minimize expected free energy
- **Variational Free Energy**: A measure of surprise that the agent seeks to minimize

### Key Mathematical Concepts

- **Variational Free Energy**: F = D_KL[q(s)||p(s|o)] - log p(o)
- **Expected Free Energy**: G = E_q(s,o)[log q(s) - log p(s,o)]
- **Belief Update**: q(s) ← argmin F[q(s)]

## Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup
├── config/                            # Configuration files
│   ├── environment_config.yaml        # Environment parameters
│   └── agent_config.yaml              # Agent parameters
├── src/                               # Source code
│   ├── __init__.py
│   ├── environment/                   # Simulation environment
│   │   ├── __init__.py
│   │   ├── navigation_env.py          # Navigation environment
│   │   └── sensors.py                 # Sensor models
│   ├── active_inference/              # Active inference implementation
│   │   ├── __init__.py
│   │   ├── generative_model.py        # Generative model
│   │   ├── recognition_model.py       # Recognition model
│   │   ├── policy_model.py            # Policy selection
│   │   └── variational_inference.py   # Variational inference
│   ├── agents/                        # Agent implementations
│   │   ├── __init__.py
│   │   ├── active_inference_agent.py  # Active inference agent
│   │   └── rl_agents.py               # RL baseline agents
│   ├── evaluation/                    # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Performance metrics
│   │   └── comparison.py              # Comparison with RL
│   └── visualization/                 # Visualization tools
│       ├── __init__.py
│       ├── belief_visualizer.py       # Belief state visualization
│       └── navigation_visualizer.py   # Navigation visualization
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_active_inference.py
│   └── test_agents.py
├── examples/                          # Example scripts
│   ├── basic_navigation.py            # Basic navigation example
│   ├── dynamic_obstacles.py           # Dynamic obstacles example
│   └── comparison_experiment.py       # RL comparison experiment
├── docs/                              # Documentation
│   ├── theory.md                      # Theoretical background
│   ├── implementation.md              # Implementation details
│   └── results.md                     # Experimental results
└── data/                              # Data and results
    ├── experiments/                   # Experiment results
    └── models/                        # Trained models
```

## Key Features

### 1. Active Inference Implementation
- **Generative Model**: Probabilistic state-space model with transition dynamics
- **Recognition Model**: Variational inference for belief updates
- **Policy Selection**: Expected Free Energy minimization for action selection
- **Online Learning**: Continuous belief updates from sensor data

### 2. Simulation Environment
- **Dynamic Navigation**: Obstacles, changing goals, and environmental dynamics
- **Sensor Models**: LIDAR, RGB camera, and IMU sensor simulations
- **Multiple Scenarios**: Various navigation challenges and environments

### 3. Evaluation Framework
- **Performance Metrics**: Success rate, time efficiency, adaptability
- **Comparison Tools**: Benchmarking against RL methods (PPO, DQN)
- **Statistical Analysis**: Confidence intervals and significance testing

### 4. Visualization
- **Belief States**: Real-time visualization of internal beliefs
- **Navigation Paths**: Trajectory visualization with uncertainty
- **Free Energy**: Monitoring of variational free energy minimization

## Research Contributions

This implementation demonstrates:

1. **Theoretical Rigor**: Faithful implementation of active inference principles
2. **Practical Applicability**: Real-world navigation scenarios
3. **Comparative Analysis**: Systematic comparison with established methods
4. **Adaptive Behavior**: Dynamic response to environmental changes
5. **Scalability**: Extensible architecture for complex scenarios

## Documentation

- [Theoretical Background](docs/theory.md)
- [Implementation Details](docs/implementation.md)
- [Experimental Results](docs/results.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is inspired by the work of Karl Friston and the active inference community. Special thanks to Professor Rish for guidance on cognitive-scientific AI approaches. # Adaptive-Robot-Navigation-System
