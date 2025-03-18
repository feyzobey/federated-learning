import serial
import logging

# Set up logging to text file
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("master_log.txt")])  # Log to a text file


def setup_uart():
    uart_port = input("Enter UART port for master device (e.g., /dev/ttyUSB2 or COM5): ")
    baud_rate = 115200

    try:
        ser = serial.Serial(uart_port, baud_rate, timeout=1)
        logging.info(f"UART Master initialized on {uart_port} at {baud_rate} baud.")
        return ser
    except Exception as e:
        logging.error(f"Error initializing UART: {e}")
        return None


def main():
    ser = setup_uart()
    if not ser:
        return

    while True:
        try:
            data = ser.readline().decode().strip()
            if data:
                logging.info(f"Received: {data}")
                ser.write(f"ACK: {data}\n".encode())  # Send acknowledgment
        except Exception as e:
            logging.error(f"UART Error: {e}")


if __name__ == "__main__":
    main()
