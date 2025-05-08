import ast
import glob
import logging
import re
from datetime import datetime

import numpy as np
import torch


def verify_shapes_match(model_updates, layer_name):
    """Verify that all clients have the same shape for a given layer"""
    shapes = []
    for update in model_updates:
        if layer_name in update["model_params"]:
            tensor = np.array(update["model_params"][layer_name])
            shapes.append(tensor.shape)

    if not shapes:
        return None

    if not all(s == shapes[0] for s in shapes):
        logging.error(f"Shape mismatch in {layer_name}: {shapes}")
        raise ValueError(f"Shape mismatch in {layer_name}")

    return shapes[0]


def get_original_shape(values, layer_name):
    """Convert flattened array back to original shape based on layer name"""
    if "conv" in layer_name.lower():
        # For conv layers, reshape to (out_channels, in_channels, kernel_size)
        if len(values) == 288:  # First conv layer
            return (32, 3, 3)
        elif len(values) == 2048:  # Second conv layer
            return (64, 32, 1)
    elif "dense" in layer_name.lower() and "bias" not in layer_name.lower():
        # For dense layers, use appropriate shapes
        if len(values) == 3200:
            return (50, 64)
        elif len(values) == 300:
            return (6, 50)
    elif "bias" in layer_name.lower():
        # For bias terms, keep as 1D
        return (len(values),)

    # If no specific shape is found, keep original flattened shape
    return (len(values),)


def process_federated_learning(iteration=1):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting federated learning process for iteration {iteration}")

    param_files = glob.glob(f"parameters/user_*.txt")
    logging.info(f"Found {len(param_files)} parameter files")

    if len(param_files) == 0:
        raise FileNotFoundError(f"No parameter files found in parameters directory")

    model_updates = []

    for file_path in param_files:
        user_id_match = re.search(r"user_(\d+)", file_path)
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

    # Verify shapes match across all clients
    first_update = model_updates[0]["model_params"]
    for layer_name in first_update.keys():
        verify_shapes_match(model_updates, layer_name)

    # --- Federated Averaging ---
    class OfflineFedServer:
        def __init__(self, model_updates):
            self.model_updates = model_updates
            self.num_clients = len(model_updates)

        def federated_averaging(self):
            avg_params = {}
            first = self.model_updates[0]["model_params"]

            # Initialize with zeros
            for layer, values in first.items():
                avg_params[layer] = torch.zeros_like(torch.tensor(values, dtype=torch.float32))

            # Sum all parameters
            for update in self.model_updates:
                for layer, values in update["model_params"].items():
                    avg_params[layer] += torch.tensor(values, dtype=torch.float32)

            # Average by dividing by number of clients
            for layer in avg_params:
                avg_params[layer] /= self.num_clients

                # Verify averaging is working correctly
                mean_before = torch.tensor([update["model_params"][layer] for update in self.model_updates]).mean(dim=0)
                mean_after = avg_params[layer]
                if not torch.allclose(mean_before, mean_after, rtol=1e-4):
                    logging.error(f"Averaging verification failed for {layer}")
                    raise ValueError(f"Federated averaging verification failed for {layer}")
                else:
                    logging.info(f"Verified averaging for {layer}")

            return avg_params

    server = OfflineFedServer(model_updates)
    avg_params = server.federated_averaging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"federated_learning_results_iter{iteration}_{timestamp}.txt"

    # Save results in the same format as input files
    with open(out_file, "w") as f:
        for layer_name, values in avg_params.items():
            # Write layer name
            f.write(f"{layer_name}\n")

            # Get the original shape
            values_np = values.detach().numpy()
            original_shape = get_original_shape(values_np, layer_name)

            # Write shape in the correct format
            f.write(f"{original_shape}\n")

            # Reshape values to original shape and write
            values_reshaped = values_np.reshape(original_shape)
            values_flat = values_reshaped.flatten()
            formatted_values = " ".join([f"{val:.6f}" for val in values_flat])
            f.write(f"{formatted_values}\n")

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
