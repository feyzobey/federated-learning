import copy
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class ActivityNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_classes=6):
        super(ActivityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def generate_dummy_data(num_clients: int, samples_per_client: int):
    activities = ["Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"]
    activity_dist = [0.386, 0.312, 0.055, 0.044, 0.112, 0.091]
    client_data = {}

    for client_id in range(num_clients):
        # generate activity labels based on distribution
        y = np.random.choice(len(activities), size=samples_per_client, p=activity_dist)

        # generate realistic accelerometer data based on activity
        X = np.zeros((samples_per_client, 3))
        for i in range(samples_per_client):
            activity = y[i]
            if activity == 0:  # Walking
                X[i] = np.random.normal([0, 0, 10], [2, 2, 1])
            elif activity == 1:  # Jogging
                X[i] = np.random.normal([0, 0, 10], [4, 4, 2])
            elif activity == 2:  # Sitting
                X[i] = np.random.normal([0, 0, 10], [0.5, 0.5, 0.5])
            elif activity == 3:  # Standing
                X[i] = np.random.normal([0, 0, 10], [0.3, 0.3, 0.3])
            elif activity == 4:  # Upstairs
                X[i] = np.random.normal([0, 0, 12], [3, 3, 2])
            else:  # Downstairs
                X[i] = np.random.normal([0, 0, 8], [3, 3, 2])

        # clip values to realistic range (-20 to 20)
        X = np.clip(X, -20, 20)

        # convert to pytorch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        client_data[client_id] = {"X": X, "y": y}

    return client_data


def load_data_from_csv(input_dir: str = "input_datas") -> Dict[int, Dict[str, torch.Tensor]]:
    activities = ["Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"]
    activity_to_idx = {activity: idx for idx, activity in enumerate(activities)}
    client_data = {}

    # Get all client CSV files
    client_files = [f for f in os.listdir(input_dir) if f.startswith("client_") and f.endswith("_data.csv")]

    for file in client_files:
        client_id = int(file.split("_")[1])
        df = pd.read_csv(os.path.join(input_dir, file))

        # Convert data to tensors
        X = torch.FloatTensor(df[["x_acceleration", "y_acceleration", "z_acceleration"]].values)
        y = torch.LongTensor([activity_to_idx[activity] for activity in df["activity"]])

        # Split into train and test (80-20 split)
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))

        client_data[client_id] = {"train": {"X": X[indices[:train_size]], "y": y[indices[:train_size]]}, "test": {"X": X[indices[train_size:]], "y": y[indices[train_size:]]}}

    return client_data


def client_update(model: nn.Module, data: Dict[str, torch.Tensor], epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(data["X"])
        loss = criterion(outputs, data["y"])
        loss.backward()
        optimizer.step()

    return model.state_dict()


def federated_averaging(client_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    averaged_state = {}

    for key in client_states[0].keys():
        averaged_state[key] = torch.zeros_like(client_states[0][key])

        for client_state in client_states:
            averaged_state[key] += client_state[key]

        averaged_state[key] = torch.div(averaged_state[key], len(client_states))

    return averaged_state


def main():
    # Hyperparameters optimized for WISDM dataset
    num_clients = 36  # Based on number of users in WISDM
    num_rounds = 50  # Reduced rounds but increased epochs per round
    local_epochs = 10  # More epochs per round for better local training
    learning_rate = 0.0005  # Lower learning rate for stability

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Open results file for writing
    with open("results/result.txt", "w") as f:
        f.write("Federated Learning Training Results\n")
        f.write("==================================\n\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Local epochs per round: {local_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n\n")

    print("Initializing Federated Learning with WISDM dataset characteristics...")
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Learning rate: {learning_rate}")

    global_model = ActivityNet()

    print("Loading client data from CSV files...")
    client_data = load_data_from_csv()

    best_accuracy = 0.0
    round_accuracies = []  # Store accuracies for all rounds
    round_times = []  # Store time taken for each round

    # Start total training time
    total_start_time = time.time()

    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")

        # Start round time
        round_start_time = time.time()

        client_states = []
        for client_id in range(num_clients):
            client_model = copy.deepcopy(global_model)
            # Use training data for local updates
            client_state = client_update(client_model, client_data[client_id]["train"], local_epochs, learning_rate)
            client_states.append(client_state)

        global_state = federated_averaging(client_states)
        global_model.load_state_dict(global_state)

        # Evaluate on test data
        global_model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for client_id in range(num_clients):
                test_data = client_data[client_id]["test"]
                outputs = global_model(test_data["X"])
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == test_data["y"]).sum().item()
                total_samples += test_data["y"].size(0)

        accuracy = total_correct / total_samples
        round_accuracies.append(accuracy)

        # Calculate round time
        round_time = time.time() - round_start_time
        round_times.append(round_time)

        # Log results to file
        with open("results/result.txt", "a") as f:
            f.write(f"\nRound {round_num + 1}:\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Time taken: {round_time:.2f} seconds\n")
            if accuracy > best_accuracy:
                f.write(f"New best accuracy achieved!\n")
            f.write("-" * 30 + "\n")

        print(f"Global Model Test Accuracy: {accuracy:.4f}")
        print(f"Time taken: {round_time:.2f} seconds")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")

    # Calculate total training time
    total_time = time.time() - total_start_time

    # Write final summary to results file
    with open("results/result.txt", "a") as f:
        f.write("\nTraining Summary:\n")
        f.write("================\n")
        f.write(f"Best accuracy achieved: {best_accuracy:.4f}\n")
        f.write(f"Average accuracy across all rounds: {sum(round_accuracies)/len(round_accuracies):.4f}\n")
        f.write(f"Accuracy in final round: {round_accuracies[-1]:.4f}\n")
        f.write(f"Accuracy improvement from first to last round: {(round_accuracies[-1] - round_accuracies[0]):.4f}\n")
        f.write(f"\nTime Statistics:\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Average time per round: {sum(round_times)/len(round_times):.2f} seconds\n")
        f.write(f"Fastest round: {min(round_times):.2f} seconds\n")
        f.write(f"Slowest round: {max(round_times):.2f} seconds\n")

    print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")
    print("Detailed results have been saved to results/result.txt")


if __name__ == "__main__":
    main()
