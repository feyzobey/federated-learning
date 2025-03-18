# Federated Learning Implementation with 7-Layer CNN
Here is a comprehensive `README.md` file for your open-source project. It covers all the necessary aspects, such as project setup, usage, and key features, to help users understand and use the project effectively.

# Federated Learning with UART Communication

## Overview
This project demonstrates the implementation of **Federated Learning** using **UART communication** between a **master device** (UART master), **client devices** (UART clients), and a **server**. The client devices perform local training, and the model updates are sent to the server for **federated averaging**. The server then updates the global model, and the system ensures communication through UART.

### Key Features:
- Federated Learning on client devices.
- UART communication between devices.
- Logging of activities in text files and model updates in JSON format.
- Model updates sent and received through UART protocol.
- Support for up to 36 clients for federated learning.

---

## Installation

### Prerequisites
- Python 3.6 or higher.
- Required Python libraries:
  - `torch`
  - `numpy`
  - `serial`
  - `logging`
  - `json`

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Hardware Requirements:
- **Master Device** (UART Master) connected to the server via UART (e.g., `COM5`, `/dev/ttyUSB0`).
- **Client Devices** (up to 36 clients) connected to the server via UART.
- **STM32L4** microcontroller or compatible device.

---

## Project Structure

The project contains the following main components:

### 1. `fl_server.py`:
This file contains the server-side federated learning code. The server receives model updates from clients via UART, performs federated averaging, and updates the global model.

### 2. `fl_client.py`:
This file contains the client-side federated learning code. The client performs local model training and sends the model updates to the server via UART.

### 3. `uart_master.py`:
This file sets up the UART communication for the master device. It listens for messages from client devices and sends acknowledgments.

### 4. `model.py`:
This file contains the definition of the `ActivityCNN` model used for federated learning.

---

## Setup Instructions

### Step 1: Configure UART Ports
Before running the scripts, ensure the correct UART ports are set for your devices (both master and client). During runtime, you will be prompted to input the appropriate UART port for the master and client devices.

### Step 2: Run the UART Master Device
First, start the UART master device to listen for communication from clients and the server.

```bash
python uart_master.py
```

### Step 3: Start the Federated Learning Server
The server waits for model updates from the client devices, performs federated averaging, and updates the global model.

```bash
python fl_server.py
```

### Step 4: Start the Federated Learning Clients
The clients train their local models and send updates to the server.

```bash
python fl_client.py
```

### Step 5: Monitor Logs
Logs are written to text files (`server_log.txt`, `client_log.txt`, `master_log.txt`) to track the process. Model updates are stored in the `.json` files (`client_updates.json`, `client_update.json`).

---

## Data Flow

1. **Client Side**:
   - Clients perform local training using the `ActivityCNN` model.
   - After training, model updates are serialized as JSON and sent to the server via UART.

2. **Server Side**:
   - The server collects updates from clients, performs **federated averaging**, and updates the global model.
   - The server writes logs and model updates in text and JSON files.

3. **UART Communication**:
   - UART communication is used to send and receive model updates between client devices and the server.
   - The master device listens to incoming data and forwards it to the server.

---

## Logging

### Log Files:
- **`master_log.txt`**: Logs for the UART master device.
- **`server_log.txt`**: Logs for the federated learning server.
- **`client_log.txt`**: Logs for the federated learning clients.

### JSON Files:
- **`client_updates.json`**: Stores model updates received by the server.
- **`client_update.json`**: Stores model updates sent by the client.

---

## Federated Learning Process

1. **Model Initialization**:
   - The server initializes a global model (`ActivityCNN`), which is updated during the federated learning process.

2. **Client Training**:
   - Each client device performs local training using a random dataset. The model is trained for one or more epochs.

3. **Model Update**:
   - After local training, the client sends the trained model's weights to the server via UART.

4. **Federated Averaging**:
   - The server receives model updates from all clients, performs federated averaging, and updates the global model.

5. **Global Model Update**:
   - The server saves the global model after federated averaging and writes logs.

---

## Troubleshooting

1. **UART Communication Issues**:
   - Ensure that the correct UART port is selected when running the script.
   - Check that all devices are properly connected and powered.

2. **Logs Not Generated**:
   - Make sure the log directory has proper write permissions.
   - Check the log files for detailed error messages.

3. **Model Updates Not Received**:
   - Verify the client's model updates are correctly formatted as JSON.
   - Ensure there is proper communication between the client and server via UART.

---

## Contributions

This is an open-source project, and contributions are welcome! Feel free to fork this project, open issues, and submit pull requests.

---

## Acknowledgments

- The federated learning concept is based on the work in distributed machine learning and privacy-preserving computing.
- This project utilizes UART communication to simulate data transfer between devices.

---
