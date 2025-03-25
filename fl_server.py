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
    def __init__(self, uart_port, baud_rate=115200, num_clients=36):
        self.model_updates = []
        self.num_clients = num_clients
        self.serial_conn = serial.Serial(uart_port, baud_rate, timeout=1)
        logging.info(f"Federated Server initialized on {uart_port}.")

    def federated_averaging(self):
        """Performs federated averaging."""
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
        """Receives updates from UART."""
        try:
            current_client = {}
            # current_layer = None
            values = []

            while True:
                data = self.serial_conn.readline().decode().strip()
                if not data:
                    continue

                # Log raw data for debugging
                logging.debug(f"Raw data received: {data}")

                if data.startswith("CLIENT:"):
                    # Start of new client data
                    if current_client:
                        # Save previous client if exists
                        self.model_updates.append(current_client)
                        logging.info(f"Received update from client ({len(self.model_updates)}/{self.num_clients})")

                        # Save to file
                        os.makedirs("client_updates", exist_ok=True)
                        client_file = f"client_updates/client_{len(self.model_updates)}.txt"
                        with open(client_file, "w") as f:
                            f.write(str(current_client))
                        logging.info(f"Saved client update to {client_file}")

                    current_client = {}
                    client_id = int(data.split(":")[1])
                    logging.info(f"Processing data for client {client_id}")

                elif data.startswith("END"):
                    # End of client data
                    if current_client:
                        self.model_updates.append(current_client)
                        logging.info(f"Received update from client ({len(self.model_updates)}/{self.num_clients})")

                        # Save to file
                        os.makedirs("client_updates", exist_ok=True)
                        client_file = f"client_updates/client_{len(self.model_updates)}.txt"
                        with open(client_file, "w") as f:
                            f.write(str(current_client))
                        logging.info(f"Saved client update to {client_file}")

                        # Reset for next client
                        current_client = {}

                elif ":" in data:
                    # Layer data with values
                    layer_name, values_str = data.split(":", 1)
                    try:
                        # Split the values string by comma and convert to floats
                        values = [float(x) for x in values_str.split(",")]
                        current_client[layer_name] = values
                        logging.debug(f"Received {len(values)} values for layer {layer_name}")
                    except ValueError as e:
                        logging.error(f"Error parsing values for layer {layer_name}: {e}")
                        logging.error(f"Invalid values string: {values_str}")

                # Check if we have all clients
                if len(self.model_updates) >= self.num_clients:
                    break

        except Exception as e:
            logging.error(f"Error receiving data: {e}")
            logging.error(f"Raw data: {data}")

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
