import numpy as np
import os
from datetime import datetime


def generate_model_data():
    # Create a dictionary to store all the model parameters
    model_data = {}

    # Generate conv1 weights (32x3x3)
    model_data["conv1.weight"] = np.random.uniform(-1, 1, (32, 3, 3)).flatten().tolist()

    # Generate conv1 bias (32)
    model_data["conv1.bias"] = np.random.uniform(-1, 1, 32).tolist()

    # Generate conv2 weights (64x3x3)
    model_data["conv2.weight"] = np.random.uniform(-1, 1, (64, 3, 3)).flatten().tolist()

    # Generate conv2 bias (64)
    model_data["conv2.bias"] = np.random.uniform(-1, 1, 64).tolist()

    # Generate dense1 weights (50x64)
    model_data["dense1.weight"] = np.random.uniform(-1, 1, (50, 64)).flatten().tolist()

    # Generate dense1 bias (50)
    model_data["dense1.bias"] = np.random.uniform(-1, 1, 50).tolist()

    # Generate dense2 weights (6x50)
    model_data["dense2.weight"] = np.random.uniform(-1, 1, (6, 50)).flatten().tolist()

    # Generate dense2 bias (6)
    model_data["dense2.bias"] = np.random.uniform(-1, 1, 6).tolist()

    return model_data


def save_model_data(model_data, client_id):
    # Get the Downloads folder path
    # downloads_path = os.path.expanduser("~/Downloads")

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_data_client_{client_id}_{timestamp}.txt"
    filepath = os.path.join(filename)

    # Write the data to file
    with open(filepath, "w") as f:
        f.write(f"CLIENT:{client_id}\n")
        for layer_name, values in model_data.items():
            # Convert all values to strings with 6 decimal places
            values_str = ",".join([f"{v:.6f}" for v in values])
            f.write(f"{layer_name}:{values_str}\n")
        f.write("END\n")

    print(f"Model data saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Generate data for client 1
    model_data = generate_model_data()
    save_model_data(model_data, 1)

    # Print shape information
    # print("\nGenerated tensor shapes:")
    # print("conv1.weight:", (32, 3, 3))
    # print("conv1.bias:", (32,))
    # print("conv2.weight:", (64, 3, 3))
    # print("conv2.bias:", (64,))
    # print("dense1.weight:", (50, 64))
    # print("dense1.bias:", (50,))
    # print("dense2.weight:", (6, 50))
    # print("dense2.bias:", (6,))
