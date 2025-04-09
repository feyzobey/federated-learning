import logging
import serial
import torch
from datetime import datetime
import serial.tools.list_ports
import os

# Set up logging to text file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("server_log.txt")],
)


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
    def __init__(self, uart_port, baud_rate=115200, num_clients=1):
        self.model_updates = []
        self.num_clients = num_clients
        self.serial_conn = serial.Serial(uart_port, baud_rate, timeout=1)
        logging.info(f"Federated Server initialized on {uart_port}.")

    def federated_averaging(self):
        if len(self.model_updates) < self.num_clients:
            logging.info(f"Waiting for more clients... ({len(self.model_updates)}/{self.num_clients})")
            return

        # Initialize average parameters with zeros
        avg_params = {}
        first_update = self.model_updates[0]

        # Initialize with zeros of correct shape
        for key, value in first_update.items():
            avg_params[key] = torch.zeros_like(torch.tensor(value))

        # Accumulate updates
        for update in self.model_updates:
            for key, value in update.items():
                avg_params[key] += torch.tensor(value) / self.num_clients

        logging.info(f"Updated Global Model at {datetime.now()}")

        # Save the averaged model
        self.save_model(avg_params)

        # Clear updates after averaging
        self.model_updates.clear()

    def receive_client_updates(self):
        buffer = ""
        current_layer = None
        current_layer_data = ""
        try:
            current_client = {
                "client_id": None,
                "model_params": {"conv1.weight": [], "conv1.bias": [], "conv2.weight": [], "conv2.bias": [], "dense1.weight": [], "dense1.bias": [], "dense2.weight": [], "dense2.bias": []},
            }

            while True:
                try:
                    # Read raw bytes
                    raw_data = self.serial_conn.readline()

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
                        # Continuation of layer data
                        current_layer_data += data

                        # Check if we've reached the end of this layer's data
                        if data.endswith(",END") or data == "END" or "END" in data:
                            # Clean up and process the layer data
                            if "END" in current_layer_data:
                                current_layer_data = current_layer_data.split("END")[0]

                            # Remove any trailing commas
                            current_layer_data = current_layer_data.rstrip(",")

                            try:
                                # Convert values to float and store in model params
                                values = []
                                for val in current_layer_data.split(","):
                                    val = val.strip()
                                    if val:  # Skip empty entries
                                        try:
                                            values.append(float(val))
                                        except ValueError as e:
                                            logging.error(f"Error parsing value in {current_layer}: {val}")
                                            # If we can detect the corrupted part, we can try to salvage what we have
                                            if "\\x00" in val or "\\x" in val:
                                                logging.warn(f"Detected binary data in value, truncating: {val[:10]}...")
                                                break
                                            raise

                                if values:
                                    current_client["model_params"][current_layer] = values
                                    logging.info(f"Successfully parsed {len(values)} values for {current_layer}")
                            except Exception as e:
                                logging.error(f"Failed to parse {current_layer} data: {e}")

                            # Reset for next layer
                            current_layer = None
                            current_layer_data = ""

                            # If this was the end marker, process the client data
                            if "END" in data:
                                if all(len(v) > 0 for v in current_client["model_params"].values()):
                                    self.model_updates.append(current_client)
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
        uart_port = input("\nEnter UART port for the server (or 'q' to quit): ")
        if uart_port.lower() == "q":
            exit(0)
        if uart_port in available_ports:
            break
        print(f"Invalid port. Please choose from: {', '.join(available_ports)}")

    try:
        server = FederatedServer(uart_port)
        logging.info("Server Running...")

        while True:
            server.receive_client_updates()
            if len(server.model_updates) >= server.num_clients:
                server.federated_averaging()
    except serial.SerialException as e:
        logging.error(f"Serial port error: {e}")
        print(f"Error: Could not open port {uart_port}. Please check if it's available and you have permission to access it.")
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        print("\nServer stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
