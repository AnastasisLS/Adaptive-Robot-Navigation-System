# Adaptive Robot Navigation System - Clean Project

## Overview
A clean, focused implementation of active inference for robot navigation research. All non-essential files have been removed to maintain only the core research functionality.

## Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Git ignore rules
├── config/                      # Configuration files
│   ├── environment_config.yaml  # Environment parameters
│   └── agent_config.yaml        # Agent parameters
├── src/                         # Core implementation
│   ├── environment/             # Navigation environment
│   ├── active_inference/        # Active inference framework
│   ├── agents/                  # Agent implementations
│   ├── evaluation/              # Evaluation and metrics
│   └── visualization/           # Visualization tools
├── examples/                    # Experiment scripts
│   ├── basic_navigation.py      # Basic navigation experiment
│   └── comparison_experiment.py # RL comparison experiment
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── theory.md               # Theoretical background
│   ├── implementation.md       # Implementation details
│   ├── results.md              # Experimental results
│   └── api.md                  # API documentation
└── data/                        # Data directories (empty)
    ├── experiments/             # For experiment results
    └── models/                  # For trained models
```

## Core Features

### 1. Active Inference Implementation
- **Generative Model**: Probabilistic state-space model
- **Recognition Model**: Variational inference for belief updates
- **Policy Selection**: Expected Free Energy minimization
- **Online Learning**: Continuous belief updates

### 2. Simulation Environment
- **Dynamic Navigation**: Obstacles, changing goals
- **Sensor Models**: LIDAR, RGB camera, IMU
- **Curriculum Learning**: Progressive difficulty
- **Collision Detection**: Robust obstacle avoidance

### 3. Evaluation Framework
- **Performance Metrics**: Success rate, efficiency, adaptability
- **Comparison Tools**: Benchmarking against RL methods
- **Comprehensive Logging**: Detailed data collection

## Quick Start

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Run Experiments
```bash
# Basic navigation experiment
python examples/basic_navigation.py

# Comparison with RL methods
python examples/comparison_experiment.py
```

## Research Focus

This clean project focuses on:
1. **Active Inference Research**: Core implementation and theory
2. **Robot Navigation**: Adaptive behavior in dynamic environments
3. **Comparative Analysis**: Evaluation against RL baselines
4. **Reproducible Experiments**: Clean, documented code

## What Was Removed

- ❌ Cloud deployment scripts and documentation
- ❌ Docker containerization files
- ❌ Temporary experiment results (can be regenerated)
- ❌ Trained model files (can be retrained)
- ❌ Build artifacts and cache files
- ❌ Deployment utilities and summaries

## Benefits

1. **Focused Research**: Only essential code and documentation
2. **Easy Setup**: Minimal dependencies and configuration
3. **Clean Codebase**: No deployment complexity
4. **Reproducible**: All experiments can be run locally
5. **Maintainable**: Simple, clear project structure

## Next Steps

1. **Run Experiments**: Execute the example scripts
2. **Analyze Results**: Study active inference behavior
3. **Extend Research**: Add new features or experiments
4. **Document Findings**: Update results and theory docs

The project is now clean, focused, and ready for active inference research! 