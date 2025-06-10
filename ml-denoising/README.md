# ML Denoising: Quantum Circuit Error Mitigation with Machine Learning

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/qBraid/ai4quantum.git&utm_source=github&redirectUrl=ml-denoising/README.md)


A Python package that uses machine learning techniques, specifically Graph Neural Networks (GNNs), to mitigate quantum circuit errors. This package implements advanced error mitigation strategies using PyTorch Geometric and Qiskit.

## Features

- **Graph Neural Network Models**: Two model variants (Simple and Robust) for quantum error mitigation
- **Rich Circuit Representation**: Converts quantum circuits to graphs with comprehensive node and edge features
- **Noise Modeling**: Implements realistic noise models based on experimental quantum hardware
- **Comprehensive Evaluation**: Built-in metrics and visualization tools for model performance analysis
- **Scalable Training**: Efficient training pipeline with early stopping and learning rate scheduling

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ml-denoising

# Install using poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from ml_denoising.model import RobustQErrorMitigationModel
from ml_denoising.data_generation import generate_circuits_and_observables
from ml_denoising.train import train_mitigator, evaluate_mitigator
import torch

# Generate synthetic data
circuits, observables, noise_values, true_values = generate_circuits_and_observables(
    num_circuits=1000,
    min_qubits=2,
    max_qubits=6,
    noise_factor=0.2
)

# Create and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobustQErrorMitigationModel(
    node_feature_size=64,
    edge_feature_size=2,
    global_feature_size=10
).to(device)

trained_model, history = train_mitigator(
    dataset=dataset,
    model=model,
    epochs=100,
    device=device
)

# Evaluate model performance
results = evaluate_mitigator(trained_model, test_dataset, device=device)
```

### Command Line Interface

```bash
# Train a model
ml-denoising-train --config config.json

# Generate training data
ml-denoising-generate --num-circuits 1000 --output-dir data/
```

## Architecture

### Model Variants

1. **SimpleQErrorMitigationModel**: A lightweight GAT-based model for basic error mitigation
2. **RobustQErrorMitigationModel**: An advanced model with residual connections and rich feature processing

### Key Components

- **Circuit to Graph Conversion**: Transforms quantum circuits into graph representations
- **Noise Models**: Realistic noise modeling based on experimental quantum devices
- **Feature Extraction**: Comprehensive feature extraction for observables and circuit properties
- **Training Pipeline**: Robust training with validation, early stopping, and checkpointing

## Examples

### Running a Complete Experiment

```python
# examples/run_experiment.py
from ml_denoising import *

# Configuration
config = {
    'num_circuits': 2000,
    'min_qubits': 2,
    'max_qubits': 8,
    'noise_factor': 0.2,
    'model_depth': 5,
    'epochs': 200
}

# Run full experiment pipeline
results = run_scaling_experiment(config)
```

### Custom Noise Models

```python
from ml_denoising.noise_modeling import get_quera_noise_model

# Create custom noise model
noise_model = get_quera_noise_model(config_quera_noise_factor=0.5)

# Use in circuit simulation
noisy_result = execute_with_noise(circuit, noise_model)
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
poetry install --with dev,docs

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/
isort src/
flake8 src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_denoising

# Run specific test file
pytest tests/test_model.py
```

## Project Structure

```
ml-denoising/
├── src/
│   └── ml_denoising/
│       ├── __init__.py
│       ├── model.py          # Neural network models
│       ├── train.py          # Training pipeline
│       ├── data_generation.py # Data generation utilities
│       ├── circuit.py        # Circuit-to-graph conversion
│       └── noise_modeling.py # Noise model implementations
├── examples/
│   └── run_experiment.py     # Example usage
├── pyproject.toml           # Poetry configuration
└── README.md
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ml_denoising,
  title={ML Denoising: Quantum Circuit Error Mitigation with Machine Learning},
  author={Kenny Heitritter},
  organization={qBraid},
  year={2024},
  url={https://github.com/your-org/ml-denoising}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support, please open an issue on GitHub or contact [kenny@qbraid.com](mailto:kenny@qbraid.com). 