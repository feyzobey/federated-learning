import os
import time
import json
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ActivityCNN(nn.Module):
    """CNN model for activity recognition with 7 layers as specified."""

    def __init__(self, input_channels=3, num_classes=6, window_size=20):
        super(ActivityCNN, self).__init__()

        # Layer 1: First convolutional layer (3, 3, 32)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)

        # Layer 2: First ReLU activation
        self.relu1 = nn.ReLU()

        # Layer 3: First max pooling layer
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 4: Second convolutional layer (32, 3, 64)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Layer 5: Second ReLU activation
        self.relu2 = nn.ReLU()

        # Layer 6: Second max pooling layer
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate flattened size
        self.flattened_size = 64 * (window_size // 4)

        # Layer 7: Final fully connected layer
        self.fc = nn.Linear(self.flattened_size, num_classes)

        # Initialize weights
        self.initialize_weights()

        # Store model configuration
        self.config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "window_size": window_size,
            "flattened_size": self.flattened_size,
            "architecture": [
                {"layer": 1, "type": "Conv1d", "params": {"in_channels": input_channels, "out_channels": 32, "kernel_size": 3}},
                {"layer": 2, "type": "ReLU", "params": {}},
                {"layer": 3, "type": "MaxPool1d", "params": {"kernel_size": 2, "stride": 2}},
                {"layer": 4, "type": "Conv1d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3}},
                {"layer": 5, "type": "ReLU", "params": {}},
                {"layer": 6, "type": "MaxPool1d", "params": {"kernel_size": 2, "stride": 2}},
                {"layer": 7, "type": "Linear", "params": {"in_features": self.flattened_size, "out_features": num_classes}},
            ],
        }

    def initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        torch.manual_seed(42)  # For reproducibility

        # Conv1 weights and bias (Layer 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        # Conv2 weights and bias (Layer 4)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        # FC weights and bias (Layer 7)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """Forward pass through the network."""
        # Input validation and reshaping
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            x = x.transpose(1, 2)

        if x.size(1) != self.config["input_channels"]:
            raise ValueError(f"Expected input to have {self.config['input_channels']} channels, got {x.size(1)}")

        # Layer 1-3: First conv block
        x = self.conv1(x)  # Layer 1
        x = self.relu1(x)  # Layer 2
        x = self.pool1(x)  # Layer 3

        # Layer 4-6: Second conv block
        x = self.conv2(x)  # Layer 4
        x = self.relu2(x)  # Layer 5
        x = self.pool2(x)  # Layer 6

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Layer 7: Final fully connected layer
        x = self.fc(x)

        return x


class Client:
    """Client class for federated learning."""

    def __init__(self, client_id: int, data: Dict[str, torch.Tensor], device: str = "cpu", batch_size: int = 32, window_size: int = 20):
        self.client_id = client_id
        self.validate_data(data)
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.model = ActivityCNN(window_size=window_size).to(device)
        self.dataset_size = len(data["X"])
        self.training_history = []

    def validate_data(self, data: Dict[str, torch.Tensor]):
        """Validate input data format and dimensions."""
        if "X" not in data or "y" not in data:
            raise ValueError("Data must contain 'X' and 'y' keys")
        if len(data["X"]) != len(data["y"]):
            raise ValueError("Features and labels must have the same length")
        if data["X"].dim() != 3:
            raise ValueError("Features must be 3-dimensional (samples, channels, window_size)")

    def train(self, epochs: int, lr: float) -> Dict[str, torch.Tensor]:
        """Train client model on local data."""
        train_dataset = TensorDataset(self.data["X"].to(self.device), self.data["y"].to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        epoch_metrics = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # Calculate epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            epoch_metrics.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

            print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, " f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        self.training_history.extend(epoch_metrics)
        return self.get_model_params()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current model on local test data."""
        self.model.eval()
        test_dataset = TensorDataset(self.data["X"].to(self.device), self.data["y"].to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        correct = 0
        total = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                predictions.extend(predicted.cpu().numpy().tolist())
                true_labels.extend(batch_y.cpu().numpy().tolist())

        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)

        return {"accuracy": accuracy, "loss": avg_loss, "predictions": predictions, "true_labels": true_labels, "total_samples": total}

    def get_dataset_size(self) -> int:
        """Return the size of client's dataset for weighted averaging."""
        return self.dataset_size

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get current local model parameters."""
        return self.model.state_dict()

    def set_model_params(self, model_params: Dict[str, torch.Tensor]):
        """Update local model with new parameters."""
        self.model.load_state_dict(model_params)


class Server:
    """Server class for federated learning."""

    def __init__(self, window_size: int, device: str = "cpu"):
        self.device = device
        self.global_model = ActivityCNN(window_size=window_size).to(device)
        self.best_accuracy = 0.0
        self.round_history = []

        # Save initial model state
        self.initial_weights = self.global_model.initialize_weights()

    def federated_averaging(self, client_params: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Perform federated averaging on client parameters."""
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]

        averaged_params = {}
        for key in client_params[0].keys():
            averaged_params[key] = torch.zeros_like(client_params[0][key], device=self.device)

            # Weighted averaging of parameters
            for client_weight, client_param in zip(weights, client_params):
                averaged_params[key] += client_weight * client_param[key].to(self.device)

        return averaged_params

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return self.global_model.state_dict()

    def update_global_model(self, averaged_params: Dict[str, torch.Tensor]):
        """Update global model with averaged parameters."""
        self.global_model.load_state_dict(averaged_params)

    def save_model(self, path: str, round_num: int, metrics: Dict):
        """Save model with metadata."""
        save_dict = {"model_state": self.global_model.state_dict(), "round": round_num, "metrics": metrics, "config": self.global_model.config, "timestamp": datetime.now().isoformat()}
        torch.save(save_dict, path)


def setup_logging(config: Dict, log_dir: str = "results") -> Dict:
    """Setup logging directories and files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    paths = {"run_dir": run_dir, "log_file": os.path.join(run_dir, "results_fl.txt"), "model_dir": os.path.join(run_dir, "models"), "config_file": os.path.join(run_dir, "config.json")}
    os.makedirs(paths["model_dir"], exist_ok=True)

    # Save configuration
    with open(paths["config_file"], "w") as f:
        json.dump(config, f, indent=2)

    return paths


def generate_synthetic_data(num_samples: int, window_size: int) -> Dict[str, torch.Tensor]:
    """Generate synthetic data for activity recognition with realistic patterns."""
    torch.manual_seed(42)  # For reproducibility

    # Generate features (X) with shape [samples, channels, window_size]
    X = torch.randn(num_samples, 3, window_size)

    # Add some structure to the data
    for i in range(num_samples):
        # Add time-dependent patterns
        t = torch.linspace(0, 1, window_size)
        X[i, 0] += 0.5 * torch.sin(2 * torch.pi * t)  # Add sine wave to first channel
        X[i, 1] += 0.3 * torch.cos(4 * torch.pi * t)  # Add cosine wave to second channel
        X[i, 2] += 0.2 * torch.randn(window_size)  # Add noise to third channel

    # Generate labels (y) with values between 0 and 5 (6 classes)
    y = torch.randint(0, 6, (num_samples,))

    return {"X": X, "y": y}


def save_synthetic_data(data: Dict[str, torch.Tensor], file_path: str = "inputs/inputs.txt"):
    """Save synthetic data in the specified format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        # Write header
        f.write("Layer Count: 2\n\n")  # X and y layers

        # Write X data (features)
        f.write("Layer 1: features\n")
        f.write("Parameter Count: 1\n")
        f.write("Weights 1:\n")
        f.write(f"Shape: {tuple(data['X'].shape)}\n")
        f.write("Values:")
        for val in data["X"].flatten().tolist():
            f.write(f" {val:.6f}")
        f.write("\n\n")

        # Write y data (labels)
        f.write("Layer 2: labels\n")
        f.write("Parameter Count: 1\n")
        f.write("Weights 1:\n")
        f.write(f"Shape: {tuple(data['y'].shape)}\n")
        f.write("Values:")
        for val in data["y"].tolist():
            f.write(f" {val}")
        f.write("\n")

    # Also save a JSON version for easier processing
    json_path = file_path.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "X_shape": list(data["X"].shape),
                "y_shape": list(data["y"].shape),
                "X_mean": float(data["X"].mean()),
                "X_std": float(data["X"].std()),
                "y_classes": len(torch.unique(data["y"])),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def main():
    # Configuration
    config = {
        "num_clients": 36,
        "num_rounds": 10,
        "local_epochs": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
        "window_size": 20,
        "samples_per_client": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Setup logging
    paths = setup_logging(config)

    # Initialize logging
    with open(paths["log_file"], "w") as f:
        f.write("Federated Learning with 7-Layer CNN Model\n")
        f.write("=====================================\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\nModel Architecture:\n")
        f.write("- Layer 1: Conv1d (3, 3, 32)\n")
        f.write("- Layer 2: ReLU\n")
        f.write("- Layer 3: MaxPool1d\n")
        f.write("- Layer 4: Conv1d (32, 3, 64)\n")
        f.write("- Layer 5: ReLU\n")
        f.write("- Layer 6: MaxPool1d\n")
        f.write("- Layer 7: Linear\n\n")

    print("Initializing Federated Learning...")

    # Initialize server
    server = Server(window_size=config["window_size"], device=config["device"])

    # Create synthetic data for testing
    print("Generating synthetic data for clients...")
    clients = []

    # Save example synthetic data
    example_data = generate_synthetic_data(config["samples_per_client"], config["window_size"])
    save_synthetic_data(example_data)
    print("Saved example synthetic data to inputs/inputs.txt")

    for client_id in range(config["num_clients"]):
        client_data = generate_synthetic_data(config["samples_per_client"], config["window_size"])
        client = Client(client_id=client_id, data=client_data, device=config["device"], batch_size=config["batch_size"], window_size=config["window_size"])
        clients.append(client)

    print(f"Initialized {len(clients)} clients with synthetic data")

    # Training statistics
    best_accuracy = 0.0
    start_time = time.time()
    round_metrics = []

    # Federated learning rounds
    for round_num in range(config["num_rounds"]):
        print(f"\nRound {round_num + 1}/{config['num_rounds']}")
        round_start = time.time()

        # 1. Distribute global model to clients
        global_params = server.get_model_params()
        for client in clients:
            client.set_model_params(global_params)

        # 2. Local training on each client
        client_params = []
        client_sizes = []
        client_metrics = []

        for client in clients:
            params = client.train(config["local_epochs"], config["learning_rate"])
            metrics = client.evaluate()
            client_params.append(params)
            client_sizes.append(client.get_dataset_size())
            client_metrics.append(metrics)

        # 3. Server aggregates models using FedAvg
        averaged_params = server.federated_averaging(client_params, client_sizes)
        server.update_global_model(averaged_params)

        # 4. Evaluate updated model on each client
        accuracies = [m["accuracy"] for m in client_metrics]
        losses = [m["loss"] for m in client_metrics]
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)
        round_time = time.time() - round_start

        # Collect round metrics
        round_metrics.append({"round": round_num + 1, "avg_accuracy": avg_accuracy, "avg_loss": avg_loss, "round_time": round_time, "client_accuracies": accuracies, "client_losses": losses})

        # Log results
        with open(paths["log_file"], "a") as f:
            f.write(f"\nRound {round_num + 1}:\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Loss: {avg_loss:.4f}\n")
            f.write(f"Time: {round_time:.2f} seconds\n")
            f.write("Client Metrics:\n")
            for i, metrics in enumerate(client_metrics):
                f.write(f"  Client {i}: Accuracy={metrics['accuracy']:.4f}, " f"Loss={metrics['loss']:.4f}\n")

            if avg_accuracy > best_accuracy:
                f.write("New best accuracy achieved!\n")
                best_accuracy = avg_accuracy
                # Save best model
                model_path = os.path.join(paths["model_dir"], f"best_model_round_{round_num+1}.pt")
                server.save_model(model_path, round_num + 1, {"accuracy": avg_accuracy, "loss": avg_loss, "round_time": round_time})

        print(f"Round {round_num + 1} completed:")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time taken: {round_time:.2f} seconds")

    # Final summary
    total_time = time.time() - start_time

    # Save final metrics
    final_metrics = {"total_time": total_time, "best_accuracy": best_accuracy, "round_metrics": round_metrics, "config": config}

    with open(os.path.join(paths["run_dir"], "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    with open(paths["log_file"], "a") as f:
        f.write("\nTraining Summary\n")
        f.write("===============\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Best accuracy achieved: {best_accuracy:.4f}\n")
        f.write(f"Results directory: {paths['run_dir']}\n")
        f.write(f"Best model saved in: {paths['model_dir']}\n")

    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    print(f"Results saved in: {paths['run_dir']}")


if __name__ == "__main__":
    main()
