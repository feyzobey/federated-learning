import serial
import torch
import json
from datetime import datetime
from collections import defaultdict
from cnn_model import ActivityCNN


class FederatedServer:
    def __init__(self, uart_port, baud_rate=115200, num_clients=36):
        self.global_model = ActivityCNN()
        self.model_updates = []
        self.num_clients = num_clients
        self.serial_conn = serial.Serial(uart_port, baud_rate, timeout=1)

    def federated_averaging(self):
        """Performs federated averaging."""
        if len(self.model_updates) < self.num_clients:
            print(f"Waiting for more clients... ({len(self.model_updates)}/{self.num_clients})")
            return

        avg_params = defaultdict(lambda: torch.zeros_like(next(iter(self.model_updates[0]["weights"].values()))))
        for update in self.model_updates:
            for key, value in update["weights"].items():
                avg_params[key] += torch.tensor(value) / len(self.model_updates)

        self.global_model.load_state_dict(avg_params)
        print(f"Updated Global Model at {datetime.now()}")
        self.model_updates.clear()

    def receive_client_updates(self):
        """Receives updates from UART."""
        try:
            data = self.serial_conn.readline().decode().strip()
            if data:
                self.model_updates.append(json.loads(data))
                print(f"Received update from client ({len(self.model_updates)}/{self.num_clients})")
        except Exception as e:
            print(f"Error receiving data: {e}")

    def save_model(self, path="global_model.pth"):
        """Saves the global model."""
        torch.save(self.global_model.state_dict(), path)
        print(f"Model saved at {path}")


if __name__ == "__main__":
    uart_port = input("Enter UART port for the server (e.g., /dev/ttyUSB0 or COM3): ")
    server = FederatedServer(uart_port)

    print("Server Running...")

    while True:
        server.receive_client_updates()
        if len(server.model_updates) >= server.num_clients:
            server.federated_averaging()
            server.save_model()
