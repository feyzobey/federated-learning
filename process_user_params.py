import ast
import glob
import json
import logging
import re
from datetime import datetime

import numpy as np
import torch


def process_federated_learning(iteration=1):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting federated learning process for iteration {iteration}")

    param_files = glob.glob(f"parameters/user_*_iter_{iteration}_params.txt")
    logging.info(f"Found {len(param_files)} parameter files for iteration {iteration}")

    if len(param_files) == 0:
        raise FileNotFoundError(f"No parameter files found for iteration {iteration}")

    model_updates = []

    for file_path in param_files:
        user_id_match = re.search(r"user_(\d+)_iter", file_path)
        if not user_id_match:
            logging.warning(f"Could not extract user ID from filename: {file_path}")
            continue

        user_id = int(user_id_match.group(1))
        model_params = {}

        current_layer = None
        current_shape = None
        values = []

        with open(file_path, "r") as f:
            for line in f:
                data = line.strip()
                if not data:
                    continue

                # Detect layer name
                if not data.startswith("(") and re.match(r"^[a-zA-Z0-9_.]+$", data):
                    if current_layer and current_shape and values:
                        try:
                            array = np.array(values, dtype=np.float32).reshape(current_shape)
                            model_params[current_layer] = array.flatten().tolist()
                            logging.info(f"[User {user_id}] Stored layer: {current_layer} with shape {current_shape}")
                        except Exception as e:
                            logging.warning(f"[User {user_id}] Failed to store layer {current_layer}: {e}")
                    else:
                        if current_layer:
                            logging.warning(f"[User {user_id}] Skipping incomplete layer {current_layer}")
                    current_layer = data
                    current_shape = None
                    values = []
                    continue

                # Detect shape
                if data.startswith("(") and current_layer:
                    try:
                        current_shape = ast.literal_eval(data)
                        values = []
                    except Exception as e:
                        logging.warning(f"[User {user_id}] Invalid shape for {current_layer}: {data} -> {e}")
                    continue

                # Collect numeric values
                if current_layer and current_shape:
                    try:
                        numbers = [float(val) for val in data.strip().split()]
                        values.extend(numbers)
                    except ValueError as e:
                        logging.warning(f"[User {user_id}] Failed to parse floats: {data} -> {e}")

        # Final layer after EOF
        if current_layer and current_shape and values:
            try:
                array = np.array(values, dtype=np.float32).reshape(current_shape)
                model_params[current_layer] = array.flatten().tolist()
                logging.info(f"[User {user_id}] Stored final layer: {current_layer}")
            except Exception as e:
                logging.warning(f"[User {user_id}] Failed to store final layer {current_layer}: {e}")

        if model_params:
            model_updates.append({"client_id": user_id, "model_params": model_params})
            logging.info(f"Added user {user_id} with {len(model_params)} layers.")
        else:
            logging.warning(f"No valid parameters found in file: {file_path}")

    logging.info(f"Loaded model parameters from {len(model_updates)} users.")

    # --- Federated Averaging ---
    class OfflineFedServer:
        def __init__(self, model_updates):
            self.model_updates = model_updates
            self.num_clients = len(model_updates)

        def federated_averaging(self):
            avg_params = {}
            first = self.model_updates[0]["model_params"]
            for layer, values in first.items():
                avg_params[layer] = torch.zeros_like(torch.tensor(values))

            for update in self.model_updates:
                for layer, values in update["model_params"].items():
                    avg_params[layer] += torch.tensor(values)

            for layer in avg_params:
                avg_params[layer] /= self.num_clients

            return avg_params

    server = OfflineFedServer(model_updates)
    avg_params = server.federated_averaging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"federated_learning_results_iter{iteration}_{timestamp}.json"

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "num_clients": len(model_updates),
        "global_model": {k: v.tolist() for k, v in avg_params.items()},
        "participating_clients": [u["client_id"] for u in model_updates],
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved federated learning result to: {out_file}")
    return out_file


def main():
    try:
        result_path = process_federated_learning(iteration=1)
        print(f"\n✅ Federated averaging complete.\n📄 Results saved to: {result_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Exception in federated learning: {e}")


if __name__ == "__main__":
    main()
