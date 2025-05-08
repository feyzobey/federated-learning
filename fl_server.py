import ast
import logging
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports
import torch


def list_available_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
        return []

    print("\nAvailable ports:")
    for port in ports:
        print(f"- {port.device}: {port.description}")
    return [port.device for port in ports]


class FederatedServer:
    def __init__(self, input_uart_port, output_uart_port, baud_rate=115200, num_clients=1):
        self.model_updates = []
        self.num_clients = num_clients
        self.input_uart_port = input_uart_port
        self.input_serial = serial.Serial(input_uart_port, baud_rate, timeout=1)
        self.output_serial = serial.Serial(output_uart_port, baud_rate, timeout=1)
        logging.info(f"Federated Server initialized - Input port: {input_uart_port}, Output port: {output_uart_port}")

    def federated_averaging(self):
        if len(self.model_updates) < self.num_clients:
            logging.info(f"Waiting for more clients... ({len(self.model_updates)}/{self.num_clients})")
            return

        avg_params = {}
        first_update = self.model_updates[0]["model_params"]

        for key, value in first_update.items():
            avg_params[key] = torch.zeros_like(torch.tensor(value))

        for client_update in self.model_updates:
            for key, value in client_update["model_params"].items():
                avg_params[key] += torch.tensor(value)

        for key in avg_params:
            avg_params[key] /= self.num_clients

        logging.info(f"Updated Global Model at {datetime.now()}")

        self.save_results_to_file(avg_params)

        self.send_to_output_device(avg_params)

        self.model_updates.clear()

    def save_results_to_file(self, avg_params):
        """Saves the federated learning results to a text file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"federated_learning_results_{timestamp}_{input_uart_port.replace('/', '')}.txt"

            with open(filename, "w") as f:
                f.write(f"Federated Learning Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of clients participated: {self.num_clients}\n\n")

                # Write client IDs that participated
                client_ids = [update["client_id"] for update in self.model_updates]
                f.write(f"Participating clients: {client_ids}\n\n")

                # Write the global model parameters
                f.write("GLOBAL MODEL PARAMETERS\n")
                f.write("=======================\n")
                for layer_name, values in avg_params.items():
                    f.write(f"{layer_name} - Shape: {values.shape}\n")
                    # Write first few and last few values as sample
                    values_list = values.tolist()
                    if isinstance(values_list, list):
                        sample_size = min(5, len(values_list))
                        f.write(f"  First {sample_size} values: {values_list[:sample_size]}\n")
                        f.write(f"  Last {sample_size} values: {values_list[-sample_size:]}\n")
                    f.write("\n")

                f.write("\nStatistics for each layer:\n")
                for layer_name, values in avg_params.items():
                    values_tensor = torch.tensor(values.tolist())
                    f.write(f"{layer_name}:\n")
                    f.write(f"  Mean: {values_tensor.mean().item():.6f}\n")
                    f.write(f"  Min: {values_tensor.min().item():.6f}\n")
                    f.write(f"  Max: {values_tensor.max().item():.6f}\n")
                    f.write(f"  Std: {values_tensor.std().item():.6f}\n\n")

            logging.info(f"Federated learning results saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving federated learning results: {e}")

    def send_to_output_device(self, model_params):
        """Sends the averaged model parameters to the second STM32 device."""
        try:
            # Format data for transmission
            data = "MODEL_UPDATE\n"
            for layer_name, values in model_params.items():
                # Convert tensor to list and format as string
                values_str = ",".join([f"{float(v):.6f}" for v in values.tolist()])
                data += f"{layer_name}:{values_str}\n"
            data += "END\n"

            # Send data
            self.output_serial.write(data.encode("utf-8"))
            logging.info("Model parameters sent to output device successfully")
        except Exception as e:
            logging.error(f"Error sending data to output device: {e}")
            raise

    def receive_client_updates(self):
        try:
            current_client = {"client_id": len(self.model_updates) + 1, "model_params": {}}
            current_layer = None
            current_shape = None
            values = []

            while True:
                raw_data = self.input_serial.readline()

                # If no more data
                if not raw_data:
                    if current_layer and current_shape and values:
                        try:
                            array = np.array(values, dtype=np.float32).reshape(current_shape)
                            current_client["model_params"][current_layer] = array
                            logging.info(f"[Client {current_client['client_id']}] Finalized layer: {current_layer}")
                        except Exception as e:
                            logging.warning(f"[Client {current_client['client_id']}] Error on final layer reshape: {e}")
                    if current_client["model_params"]:
                        self.model_updates.append(current_client)
                    break

                # Decode the line
                try:
                    data = raw_data.decode("latin-1").strip()
                except UnicodeDecodeError:
                    data = raw_data.decode("utf-8", errors="replace").strip()

                if not data:
                    continue

                # If it's a shape line
                if data.startswith("(") and current_layer:
                    try:
                        current_shape = ast.literal_eval(data)
                        values = []
                    except Exception as e:
                        logging.warning(f"Failed to parse shape for {current_layer}: {data} -> {e}")
                    continue

                # If it's a new layer name
                if not data.startswith("(") and re.match(r"^[a-zA-Z0-9_.]+$", data):
                    if current_layer and current_shape and values:
                        try:
                            array = np.array(values, dtype=np.float32).reshape(current_shape)
                            current_client["model_params"][current_layer] = array
                            logging.info(f"[Client {current_client['client_id']}] Stored layer: {current_layer} with shape {current_shape}")
                        except Exception as e:
                            logging.warning(f"[Client {current_client['client_id']}] Failed to store layer {current_layer}: {e}")
                    elif current_layer:
                        logging.warning(f"[Client {current_client['client_id']}] Skipped incomplete layer {current_layer}")
                    current_layer = data
                    current_shape = None
                    values = []
                    continue

                # If it's a data line
                if current_layer and current_shape:
                    try:
                        values.extend([float(x) for x in data.strip().split()])
                    except ValueError as e:
                        logging.warning(f"Failed to parse floats in line: {data} -> {e}")

        except Exception as e:
            logging.error(f"Error in receive_client_updates: {e}")
            logging.error(f"Raw data: {raw_data if 'raw_data' in locals() else 'No raw data'}")
            raise


if __name__ == "__main__":
    available_ports = list_available_ports()
    if not available_ports:
        print("No ports available. Please check your connections.")
        exit(1)

    while True:
        input_uart_port = input("\nEnter UART port for receiving data (or 'q' to quit): ")
        if input_uart_port.lower() == "q":
            exit(0)
        if input_uart_port in available_ports:
            break
        print(f"Invalid port. Please choose from: {', '.join(available_ports)}")

    while True:
        output_uart_port = input("\nEnter UART port for sending data (or 'q' to quit): ")
        if output_uart_port.lower() == "q":
            exit(0)
        if output_uart_port in available_ports:
            break
        print(f"Invalid port. Please choose from: {', '.join(available_ports)}")

    try:
        server = FederatedServer(input_uart_port, output_uart_port)
        logging.info("Server Running...")

        while True:
            server.receive_client_updates()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"received_model_data_client_{server.model_updates[0]['client_id']}_{timestamp}.txt"

            # with open(filename, "w") as f:
            #     for update in server.model_updates:
            #         f.write(f"CLIENT:{update['client_id']}\n")
            #         for layer, values in update["model_params"].items():
            #             f.write(f"{layer}:{','.join(f'{value:.6f}' for value in values)}\n")
            #         f.write("END\n")
            # logging.info(f"Model data written to {filename}")

            if len(server.model_updates) >= server.num_clients:
                server.federated_averaging()
                break
    except serial.SerialException as e:
        logging.error(f"Serial port error: {e}")
        print(f"Error: Could not open port {input_uart_port} or {output_uart_port}. Please check if they're available and you have permission to access them.")
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        print("\nServer stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
