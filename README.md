# Federated Learning Implementation with 7-Layer CNN

This is an implementation of Federated Learning using the Federated Averaging (FedAvg) algorithm with a 7-layer CNN model. The implementation is designed for activity recognition tasks and includes comprehensive logging and model tracking.

## Overview

The implementation includes:
- A 7-layer CNN model for activity recognition
- Federated learning with multiple clients
- Weighted federated averaging (FedAvg)
- Comprehensive logging and metrics tracking
- Model checkpointing and experiment tracking
- Synthetic data generation with realistic patterns

## Requirements

```bash
torch>=2.1.0
numpy>=1.24.3
```

## Model Architecture

The CNN model consists of 7 layers:
1. Conv1d layer (3 input channels, 32 output channels, kernel size 3)
2. ReLU activation
3. MaxPool1d layer (kernel size 2, stride 2)
4. Conv1d layer (32 input channels, 64 output channels, kernel size 3)
5. ReLU activation
6. MaxPool1d layer (kernel size 2, stride 2)
7. Linear layer (flattened input → 6 classes)

## Implementation Details

### Federated Learning Process
- Global model initialization with Xavier weight initialization
- For each round:
  1. Global model distribution to clients
  2. Local training on client data
  3. Model aggregation using weighted FedAvg
  4. Performance evaluation on all clients

### Features
- Data validation and input checking
- Comprehensive logging and metrics tracking
- Model checkpointing and experiment organization
- Synthetic data generation with realistic patterns
- JSON export of metrics and configurations
- GPU support

### Logging and Metrics
- Detailed per-round metrics
- Client-specific performance tracking
- Training history and model checkpoints
- Configuration tracking and experiment organization
- JSON format metrics for easy analysis

## Project Structure

```
.
├── federated_learning_cnn.py  # Main implementation
├── results/                   # Results directory
│   └── run_TIMESTAMP/        # Individual run results
│       ├── results_fl.txt    # Detailed training log
│       ├── config.json       # Run configuration
│       ├── final_metrics.json# Complete training metrics
│       └── models/           # Model checkpoints
└── inputs/                   # Input data
    ├── inputs.txt           # Human-readable format
    └── inputs.json          # Machine-readable format
```

## Usage

```bash
python federated_learning_cnn.py
```

### Configuration Parameters

The following parameters can be modified in the configuration dictionary:
- `num_clients`: Number of clients (default: 36)
- `num_rounds`: Number of federated learning rounds (default: 10)
- `local_epochs`: Number of local training epochs (default: 5)
- `learning_rate`: Learning rate for local training (default: 0.001)
- `batch_size`: Batch size for training (default: 64)
- `window_size`: Input window size (default: 20)
- `samples_per_client`: Number of samples per client (default: 100)

## Results and Logging

The implementation provides comprehensive logging and metrics tracking:
- Training configuration and hyperparameters
- Model architecture details
- Per-round accuracy and loss metrics
- Client-specific performance metrics
- Training time statistics
- Best model checkpoints

Results are organized by timestamp for easy tracking and comparison:
- Human-readable logs in text format
- Machine-readable metrics in JSON format
- Model checkpoints for best performing models
- Configuration files for reproducibility

## Performance Metrics

The implementation tracks various metrics:
- Average accuracy across clients
- Individual client accuracies
- Training and evaluation loss
- Time statistics (per round, total)
- Best model performance

## Future Improvements

Potential areas for enhancement:
- Client data heterogeneity simulation
- Advanced client selection strategies
- Privacy-preserving mechanisms
- Adaptive learning rate scheduling
- Cross-client validation
- Distributed training support 