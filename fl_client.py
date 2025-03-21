import logging
import torch
import torch.optim as optim
import serial
import json
import numpy as np
from cnn_model import ActivityCNN


# Set up logging to text file
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("client_log.txt")])  # Log to a text file


class FederatedClient:
    def __init__(self, uart_port, baud_rate=115200):
        self.model = ActivityCNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.serial_conn = serial.Serial(uart_port, baud_rate, timeout=1)
        logging.info(f"Federated Client initialized on {uart_port}.")

    def train_local_model(self, data, labels, epochs=1):
        """Simulates local training."""
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(torch.tensor(data, dtype=torch.float32))
            loss = self.criterion(outputs, torch.tensor(labels, dtype=torch.float32))
            loss.backward()
            self.optimizer.step()

    def send_model_update(self):
        """Sends trained model weights via UART."""
        update = {"weights": {k: v.tolist() for k, v in self.model.state_dict().items()}, "size": 100}
        self.serial_conn.write((json.dumps(update) + "\n").encode())
        logging.info("Sent model update to server.")

        # Write JSON data to .json file
        with open("client_update.json", "a") as json_file:
            json.dump(update, json_file)
            json_file.write("\n")

    def run(self):
        """Runs training and sends updates."""
        while True:
            fake_data = np.random.rand(20, 3)
            fake_labels = np.zeros((6,))
            fake_labels[np.random.randint(0, 6)] = 1
            self.train_local_model(fake_data, fake_labels)
            self.send_model_update()


if __name__ == "__main__":
    uart_port = input("Enter UART port for the client (e.g., /dev/ttyUSB1 or COM4): ")
    client = FederatedClient(uart_port)
    client.run()
