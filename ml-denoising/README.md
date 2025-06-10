# ML Denoising: Quantum Circuit Error Mitigation with Machine Learning

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/qBraid/ai4quantum.git&utm_source=github&redirectUrl=/ml-denoising/README.md)


A Python package that uses machine learning techniques, specifically Graph Neural Networks (GNNs), to mitigate quantum circuit errors. This package implements advanced error mitigation strategies using PyTorch Geometric and Qiskit.

## Features

- **Graph Neural Network Models**: Simple model for quantum error mitigation
- **Rich Circuit Representation**: Converts quantum circuits to graphs that are fed into pytorch geometric
- **Noise Modeling**: Implements realistic noise models based on experimental quantum hardware
- **Comprehensive Evaluation**: Built-in metrics and visualization tools for model performance analysis
- **Scalable Training**: Efficient training pipeline with early stopping and learning rate scheduling

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/qBraid/ai4quantum.git
cd ai4quantum/ml-denoising/

# Install using poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
pip install -e .
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
  year={2025},
  url={https://github.com/qbraid/ai4quantum/ml-denoising}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support, please open an issue on GitHub or contact [kenny@qbraid.com](mailto:kenny@qbraid.com). 