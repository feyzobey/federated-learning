import serial

BAUD_RATE = 115200


def request_model_update(uart_port):
    """Request model updates from the STM32 Slave device."""
    try:
        with serial.Serial(uart_port, BAUD_RATE, timeout=2) as uart:
            uart.write(b"GET_MODEL_UPDATE\n")  # Send request
            response = uart.readline().decode().strip()
            print(f"Received from STM32: {response}")
            return response  # Model parameters as a JSON string
    except Exception as e:
        print(f"UART Error: {e}")


if __name__ == "__main__":
    print("Requesting model update from STM32...")
    uart_port = input("Enter UART port: ")
    update = request_model_update(uart_port)
    if update:
        print("Model Update Received:", update)
