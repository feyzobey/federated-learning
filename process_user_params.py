import os
import re
import torch
import json
import logging
import glob
from datetime import datetime
import fl_server

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("fedavg_process.log", mode="w")])


def process_federated_learning(num_users=36, iteration=1):
    """
    Processes the federated learning for the specified iteration
    1. Reads all user parameter files for the specified iteration
    2. Uses fl_server for federated averaging
    3. Saves results as JSON file
    """
    logging.info(f"Starting federated learning process for iteration {iteration}")

    # Find all parameter files for the specified iteration
    pattern = f"parameters/user_*_iter_{iteration}_params.txt"
    param_files = glob.glob(pattern)
    logging.info(f"Found {len(param_files)} parameter files for iteration {iteration}")

    if len(param_files) == 0:
        raise FileNotFoundError(f"No parameter files found for iteration {iteration}")

    # Layer names to extract from parameter files
    layers = ["conv1d.weight", "conv1d.bias", "conv1d_1.bias", "dense.weight", "dense.bias", "dense_1.weight", "dense_1.bias"]

    # Collect model updates from all users
    model_updates = []

    # Process each parameter file
    for file_path in param_files:
        # Extract user ID from filename
        user_id_match = re.search(r"user_(\d+)_iter", file_path)
        if not user_id_match:
            logging.error(f"Could not extract user ID from filename: {file_path}")
            continue

        user_id = int(user_id_match.group(1))
        model_params = {}

        # Read and parse parameter file
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Parse each layer's parameters
            for layer in layers:
                pattern = f"{layer}:(.*?)(?=(?:{'|'.join(layers)}|\Z))"
                match = re.search(pattern, content, re.DOTALL)

                if match:
                    values_text = match.group(1).strip()
                    # Split by whitespace and convert to floats
                    values = [float(val) for val in values_text.split() if val.strip()]
                    model_params[layer] = values
                else:
                    logging.warning(f"Could not find layer {layer} in file {file_path}")
                    model_params[layer] = []

            # Add to model updates if all layers were parsed successfully
            if all(len(vals) > 0 for vals in model_params.values()):
                model_updates.append({"client_id": user_id, "model_params": model_params})
                logging.info(f"Loaded parameters for user {user_id}")
            else:
                logging.warning(f"Skipped file {file_path} due to missing layer data")

        except Exception as e:
            logging.error(f"Error parsing file {file_path}: {e}")

    logging.info(f"Successfully loaded parameters for {len(model_updates)} users")

    # Create a modified version of FederatedServer for offline processing
    class OfflineFedServer:
        def __init__(self, model_updates):
            self.model_updates = model_updates
            self.num_clients = len(model_updates)

        def federated_averaging(self):
            """Implements federated averaging from fl_server.py"""
            # Initialize average parameters with zeros
            # Initialize average parameters with zeros
            avg_params = {}
            first_update = self.model_updates[0]["model_params"]

            # Initialize with zeros of correct shape
            for key, value in first_update.items():
                avg_params[key] = torch.zeros_like(torch.tensor(value))

            # First accumulate all updates (sum)
            for client_update in self.model_updates:
                for key, value in client_update["model_params"].items():
                    avg_params[key] += torch.tensor(value)

            # Then divide by number of clients (average)
            for key in avg_params:
                avg_params[key] /= self.num_clients

            logging.info(f"Updated Global Model at {datetime.now()}")
            return avg_params

    # Perform federated averaging
    server = OfflineFedServer(model_updates)
    avg_params = server.federated_averaging()

    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"federated_learning_results_iter{iteration}_{timestamp}.json"

    # Prepare data for JSON serialization (convert tensors to lists)
    json_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "num_clients": len(model_updates),
        "participating_clients": [update["client_id"] for update in model_updates],
        "global_model": {k: v.tolist() for k, v in avg_params.items()},
        "statistics": {
            k: {
                "mean": float(torch.tensor(v.tolist()).mean().item()),
                "min": float(torch.tensor(v.tolist()).min().item()),
                "max": float(torch.tensor(v.tolist()).max().item()),
                "std": float(torch.tensor(v.tolist()).std().item()),
            }
            for k, v in avg_params.items()
        },
    }

    # Save as JSON file
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info(f"Federated learning results saved as JSON to {json_filename}")

    # Also save model parameters in PyTorch format
    model_filename = f"global_model_iter{iteration}_{timestamp}.pth"
    torch.save(avg_params, model_filename)
    logging.info(f"Model parameters saved at {model_filename}")

    return json_filename, model_filename


def main():
    try:
        # Process parameters for 36 users, first iteration
        json_file, model_file = process_federated_learning(num_users=36, iteration=1)

        print(f"Federated averaging completed successfully.")
        print(f"Results saved to: {json_file}")
        print(f"Model saved to: {model_file}")

    except Exception as e:
        logging.error(f"Error in processing: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
