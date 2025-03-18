### this part of process will be in the STM32 ###
import serial

# Configure the UART port (adjust port and baudrate as needed)
ser = serial.Serial(port="/dev/ttyUSB1", baudrate=115200, timeout=1)


def read_data():
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            print(f"Received from Master: {data}")
            response = f"Ack: {data}"  # Example response
            ser.write(response.encode())  # Send acknowledgment


if __name__ == "__main__":
    read_data()
