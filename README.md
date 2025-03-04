# Federated Learning Implementation with FedAvg

This is a simple implementation of Federated Learning using the Federated Averaging (FedAvg) algorithm. The implementation is inspired by the WISDM activity recognition dataset structure but uses dummy data for demonstration purposes.

## Overview

The implementation includes:
- A simple neural network model for activity recognition
- Simulated client data distribution
- FedAvg implementation
- Training loop with multiple clients

## Requirements

```bash
numpy==1.24.3
torch==2.1.0
scikit-learn==1.3.0
pandas==2.0.3
```

## Implementation Details

1. **Model Architecture**: A simple feed-forward neural network with 3 layers
   - Input: 3 features (x, y, z accelerometer data)
   - Hidden layer: 64 units with ReLU activation
   - Output: 6 classes (Walking, Jogging, Sitting, Standing, Upstairs, Downstairs)

2. **Federated Learning Process**:
   - The global model is initialized
   - For each round:
     - Each client receives a copy of the global model
     - Clients train the model on their local data
     - Client models are aggregated using FedAvg
     - The global model is updated with the aggregated parameters

3. **Data Simulation**:
   - Generates dummy accelerometer data for each client
   - Similar structure to WISDM dataset
   - Random distribution of activities

## Usage

```bash
python federated_learning.py
```

The script will run the federated learning process for the specified number of rounds and display the global model accuracy after each round.

## Parameters

You can modify these parameters in the `main()` function:
- `num_clients`: Number of clients participating in federated learning
- `num_rounds`: Number of federated learning rounds
- `local_epochs`: Number of training epochs for each client
- `learning_rate`: Learning rate for local training
- `samples_per_client`: Number of samples per client 