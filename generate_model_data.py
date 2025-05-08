import numpy as np
import os
from datetime import datetime


def generate_model_data():
    # Create a dictionary to store all the model parameters
    model_data = {}
    model_shapes = {}

    # Generate conv1d weights (32x3x3)
    model_data["conv1d.weight"] = np.random.uniform(-1, 1, (32, 3, 3))
    model_shapes["conv1d.weight"] = (32, 3, 3)

    # Generate conv1d bias (32)
    model_data["conv1d.bias"] = np.random.uniform(-1, 1, (32,))
    model_shapes["conv1d.bias"] = (32,)

    # Generate conv1d_1 weights (64x32x3)
    model_data["conv1d_1.weight"] = np.random.uniform(-1, 1, (64, 32, 3))
    model_shapes["conv1d_1.weight"] = (64, 32, 3)

    # Generate conv1d_1 bias (64)
    model_data["conv1d_1.bias"] = np.random.uniform(-1, 1, (64,))
    model_shapes["conv1d_1.bias"] = (64,)

    # Generate dense weights (50x64)
    model_data["dense.weight"] = np.random.uniform(-1, 1, (50, 64))
    model_shapes["dense.weight"] = (50, 64)

    # Generate dense bias (50)
    model_data["dense.bias"] = np.random.uniform(-1, 1, (50,))
    model_shapes["dense.bias"] = (50,)

    # Generate dense_1 weights (6x50)
    model_data["dense_1.weight"] = np.random.uniform(-1, 1, (6, 50))
    model_shapes["dense_1.weight"] = (6, 50)

    # Generate dense_1 bias (6)
    model_data["dense_1.bias"] = np.random.uniform(-1, 1, (6,))
    model_shapes["dense_1.bias"] = (6,)

    return model_data, model_shapes


def save_model_data(model_data, model_shapes):
    # write text file with format like parameters/user_1_iter_1_params.txt

    filename = f"dense_model_data.txt"
    # make flatten the model data
    flattened_model_data = {}
    for layer_name, values in model_data.items():
        flattened_model_data[layer_name] = values.flatten()

    with open(filename, "w") as f:
        for layer_name, values in flattened_model_data.items():
            f.write(f"{layer_name}\n")

    # write the flattened model data to the file
    with open(filename, "w") as f:
        for layer_name, values in flattened_model_data.items():
            f.write(f"{layer_name}\n")
            f.write(f"{model_shapes[layer_name]}\n")
            for value in values:
                f.write(f"{value:.6f} ")
            f.write("\n")


if __name__ == "__main__":
    # Generate data
    model_data, model_shapes = generate_model_data()

    # Save to parameters directory with the expected filename
    save_model_data(model_data, model_shapes)
