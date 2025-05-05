import logging
import serial
import torch
from datetime import datetime
import serial.tools.list_ports
import re

# Set up logging to text file
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("server_log.txt")])


def list_available_ports():
    """List all available serial ports."""
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
        # Connection for receiving data from first STM32
        self.input_serial = serial.Serial(input_uart_port, baud_rate, timeout=1)
        # Connection for sending data to second STM32
        self.output_serial = serial.Serial(output_uart_port, baud_rate, timeout=1)
        logging.info(f"Federated Server initialized - Input port: {input_uart_port}, Output port: {output_uart_port}")

    def federated_averaging(self):
        if len(self.model_updates) < self.num_clients:
            logging.info(f"Waiting for more clients... ({len(self.model_updates)}/{self.num_clients})")
            return

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

        # Save the averaged model
        self.save_model(avg_params)

        # Save the federated learning results to a text file
        self.save_results_to_file(avg_params)

        # Send the averaged model to the second STM32 device
        self.send_to_output_device(avg_params)

        # Clear updates after averaging
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
        buffer = ""
        current_layer = None
        current_layer_data = ""
        try:
            current_client = {
                "client_id": None,
                "model_params": {"conv1.weight": [], "conv1.bias": [], "conv2.weight": [], "conv2.bias": [], "dense1.weight": [], "dense1.bias": [], "dense2.weight": [], "dense2.bias": []},
            }

            # Define pattern to detect layer markers in the data
            layer_pattern = re.compile(r"(conv\d+\.weight|conv\d+\.bias|dense\d+\.weight|dense\d+\.bias):")

            while True:
                try:
                    # Read raw bytes
                    raw_data = self.input_serial.readline()

                    # Decode the data
                    try:
                        data = raw_data.decode("latin-1").strip()
                    except UnicodeDecodeError:
                        data = raw_data.decode("utf-8", errors="replace").strip()

                    if not data:
                        continue

                    # Handle new layer data or continuation of existing layer
                    if ":" in data and not current_layer:
                        # This is potentially the start of layer data
                        parts = data.split(":", 1)
                        layer_name = parts[0]

                        if layer_name in current_client["model_params"]:
                            current_layer = layer_name
                            current_layer_data = parts[1]
                        else:
                            # Could be client info or other metadata
                            if data.startswith("CLIENT:"):
                                current_client["client_id"] = int(data.split(":")[1])
                            buffer += data + "\n"
                    elif current_layer:
                        # Check if another layer marker appears in this data
                        layer_match = layer_pattern.search(current_layer_data + data)

                        if layer_match and layer_match.group(1) != current_layer:
                            # Found a new layer marker in the middle of current layer data
                            # Instead of using rfind, get all data up to the new layer marker
                            split_point = layer_match.start()

                            # Extract all data up to the new layer marker
                            valid_data = (current_layer_data + data)[:split_point]

                            # Clean up the data - remove any "END" markers and trailing/leading whitespace
                            valid_data = valid_data.replace("END", "").strip()

                            try:
                                # Convert values to float and store in model params
                                values = []
                                # Split by comma and process all non-empty values
                                raw_values = [v.strip() for v in valid_data.split(",") if v.strip()]

                                for val in raw_values:
                                    try:
                                        values.append(float(val))
                                    except ValueError as e:
                                        if "\\x00" not in val and "\\x" not in val:
                                            logging.error(f"Error parsing value in {current_layer}: {val}")

                                if values:
                                    current_client["model_params"][current_layer] = values.copy()
                                    logging.info(f"Successfully parsed {len(values)} values for {current_layer}")
                                    logging.debug(f"First value: {values[0]}, Last value: {values[-1]}")

                            except Exception as e:
                                logging.error(f"Failed to parse {current_layer} data: {e}")

                            # Setup for the new layer we detected
                            new_layer_start = layer_match.start()
                            remaining_data = (current_layer_data + data)[new_layer_start:]
                            parts = remaining_data.split(":", 1)
                            current_layer = parts[0]
                            current_layer_data = parts[1] if len(parts) > 1 else ""
                        else:
                            # Normal continuation of layer data
                            current_layer_data += data

                            # Check if we've reached the end of this layer's data
                            if data.endswith(",END") or data == "END" or "END" in data:
                                # Clean up and process the layer data
                                if "END" in current_layer_data:
                                    current_layer_data = current_layer_data.split("END")[0]

                                # Fix: Ensure trailing comma is present but don't lose data if no comma
                                if not current_layer_data.endswith(","):
                                    current_layer_data += ","

                                try:
                                    # Convert values to float and store in model params
                                    values = []
                                    # Split by comma, but filter out empty entries that might come from trailing comma
                                    for val in [v for v in current_layer_data.split(",") if v.strip()]:
                                        val = val.strip()
                                        try:
                                            values.append(float(val))
                                        except ValueError as e:
                                            logging.error(f"Error parsing value in {current_layer}: {val}")
                                            # If we can detect the corrupted part, we can try to salvage what we have
                                            if "\\x00" in val or "\\x" in val:
                                                logging.warning(f"Detected binary data in value, truncating: {val[:10]}...")
                                                continue  # Skip this value instead of breaking

                                            # Check if this value contains another layer marker
                                            layer_in_val = layer_pattern.search(val)
                                            if layer_in_val:
                                                logging.warning(f"Found layer marker in value: {val}")
                                                # Just skip this value
                                                continue

                                            # If no recovery is possible, just skip this problematic value
                                            logging.warning(f"Skipping problematic value: {val}")
                                            continue

                                    if values:
                                        current_client["model_params"][current_layer] = values
                                        logging.info(f"Successfully parsed {len(values)} values for {current_layer}")
                                        # Log to verify we're getting all values
                                        logging.debug(f"Layer {current_layer} values: {values}")
                                except Exception as e:
                                    logging.error(f"Failed to parse {current_layer} data: {e}")

                                # Reset for next layer
                                current_layer = None
                                current_layer_data = ""

                                # If this was the end marker, process the client data
                                if "END" in data:
                                    if all(len(v) > 0 for v in current_client["model_params"].values()):
                                        # Before adding to model_updates, log the lengths of each layer
                                        for layer_name, layer_values in current_client["model_params"].items():
                                            logging.info(f"Final layer {layer_name} has {len(layer_values)} values")

                                        # Create a deep copy to prevent reference issues
                                        client_copy = {"client_id": current_client["client_id"], "model_params": {k: list(v) for k, v in current_client["model_params"].items()}}
                                        self.model_updates.append(client_copy)

                                        logging.info(f"Received complete update from client {current_client['client_id']}")
                                        current_client = {
                                            "client_id": None,
                                            "model_params": {
                                                "conv1.weight": [],
                                                "conv1.bias": [],
                                                "conv2.weight": [],
                                                "conv2.bias": [],
                                                "dense1.weight": [],
                                                "dense1.bias": [],
                                                "dense2.weight": [],
                                                "dense2.bias": [],
                                            },
                                        }
                                    buffer = ""
                    else:
                        # Other data not part of a layer
                        buffer += data + "\n"

                        # Check if we've reached the end marker
                        if "END" in buffer:
                            # If we have an end marker without processing layers properly,
                            # it's likely a format issue
                            logging.info("Received END marker, finalizing client update")
                            buffer = ""

                    # Check if we have all clients
                    if len(self.model_updates) >= self.num_clients:
                        break

                except ValueError as e:
                    logging.error(f"Error parsing value: {e}")
                    # Don't lose the current layer progress
                    if current_layer:
                        logging.error(f"Error occurred while parsing {current_layer}")
                        current_layer = None
                        current_layer_data = ""
                except Exception as e:
                    logging.error(f"Error processing data: {e}")
                    logging.error(f"Raw data: {raw_data}")

        except Exception as e:
            logging.error(f"Error in receive_client_updates: {e}")
            raise

    def save_model(self, model_params, path="global_model.pth"):
        """Saves the global model parameters."""
        torch.save(model_params, path)
        logging.info(f"Model parameters saved at {path}")


if __name__ == "__main__":
    # List available ports first
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
            # write the received data to a file as same format model_data_client_1_20250408_162929.txt
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"received_model_data_client_{server.model_updates[0]['client_id']}_{timestamp}.txt"

            with open(filename, "w") as f:
                for update in server.model_updates:
                    f.write(f"CLIENT:{update['client_id']}\n")
                    for layer, values in update["model_params"].items():
                        f.write(f"{layer}:{','.join(f'{value:.6f}' for value in values)}\n")
                    f.write("END\n")
            logging.info(f"Model data written to {filename}")

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
