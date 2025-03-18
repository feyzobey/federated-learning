import sys
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QComboBox, 
                             QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLabel, 
                             QLineEdit, QMessageBox ,QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6 import uic
from about_dialog import AboutDialog
import webbrowser
class SerialReaderThread(QThread):
    data_received = pyqtSignal(bytes)
    connection_status = pyqtSignal(bool)

    def _init_(self, port, baudrate):
        super()._init_()
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.running = True

    def run(self):
        try:
            self.serial_port = serial.Serial(self.port, self.baudrate)
            self.connection_status.emit(True)
            while self.running:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(1000)
                    self.data_received.emit(data)
                QThread.msleep(10)
        except serial.SerialException as e:
            self.connection_status.emit(False)
            self.data_received.emit(bytes(f"Serial error: {e}", 'utf-8'))
        except Exception as e:
            self.data_received.emit(bytes(f"Serial error: {e}", 'utf-8'))

    def stop(self):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

    def write_data(self, data):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(data)

class SerialMonitor(QMainWindow):
    def _init_(self):
        super()._init_()

        
        self.resize(800, 600)

        self.serial_thread = None

        self.initUI()

    def initUI(self):
        uic.loadUi("ui/main_window.ui",self)
        self.setWindowTitle('Serial Monitor - Quark Optical')
        self.refresh_ports()

        self.baudrate_selector.setText("115200")
        self.base_selector.addItems(["Decimal", "Hexadecimal", "Binary","String"])

        self.text_edit.setReadOnly(True)
        
        self.send_button.clicked.connect(self.send_data)
        self.send_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_reading)

        self.stop_button.clicked.connect(self.stop_reading)
        self.stop_button.setEnabled(False)

        self.refresh_button.clicked.connect(self.refresh_ports)
        
        self.aaaa.setStyleSheet("background-color: #096bb2;")
        
        self.infoButton.clicked.connect(self.show_info)

        self.clear_button.clicked.connect(self.text_edit.clear)
        self.send_button_2.clicked.connect(self.send_data2)

        self.browse_button.clicked.connect(self.open_file_dialog)

        self.send_button_3.clicked.connect(self.send_bin_file)

    def open_file_dialog(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "", "Tüm Dosyalar ();;Bin Dosyaları (.bin)", options=options)
        self.file = file_path
        if file_path:
            self.file_path.setText(file_path)

    def show_info(self):
        info = AboutDialog(parent=self)
        if info.exec():
            pass
    
    def refresh_ports(self):
        self.port_selector.clear()
        self.port_selector.addItems(self.list_serial_ports())

    def list_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def start_reading(self):
        if self.serial_thread is not None:
            self.serial_thread.stop()

        port = self.port_selector.currentText()
        try:
            baudrate = int(self.baudrate_selector.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Baudrate must be an integer.")
            return
        

        self.serial_thread = SerialReaderThread(port, baudrate)
        self.serial_thread.data_received.connect(self.display_data)


        self.serial_thread.connection_status.connect(self.handle_connection_status)
        self.serial_thread.start()

    def stop_reading(self):
        if self.serial_thread is not None:
            self.serial_thread.stop()
            self.serial_thread = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.send_button.setEnabled(False)

    def display_data(self, data):
        
        base = self.base_selector.currentText()
        if base == "Decimal":
            text = ' '.join(str(byte) for byte in data)
        elif base == "Hexadecimal":
            text = ' '.join(f"{byte:02X}" for byte in data)
        elif base == "Binary":
            text = ' '.join(f"{byte:08b}" for byte in data)
        elif base=="String":
            text=str(data)
        self.text_edit.append(text)

    def send_data(self):
        if self.serial_thread and self.serial_thread.serial_port and self.serial_thread.serial_port.is_open:

            data = self.send_line_edit.text()
            
            self.serial_thread.write_data(data.encode())
            
    def send_data2(self):
        if self.serial_thread and self.serial_thread.serial_port and self.serial_thread.serial_port.is_open:
            base = self.base_selector.currentText()
            input_text = self.send_line_edit_2.text()

            try:
                data = bytes(int(input_text))
                self.serial_thread.write_data(data)
            except ValueError as e:
                QMessageBox.critical(self, "Input Error", f"Invalid data format: {e}")

    def send_bin_file(self):
        if self.serial_thread and self.serial_thread.serial_port and self.serial_thread.serial_port.is_open:
            try:
                with open(self.file, 'rb') as file:
                    data = file.read()
                    print(data)
                    self.serial_thread.write_data(data)
                    
            except Exception as e:
                QMessageBox.critical(self, "Input Error", f"Invalid file: {e}")


    def handle_connection_status(self, status):
        if status:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.send_button.setEnabled(True)
        else:
            QMessageBox.critical(self, "Connection Error", "Failed to connect to the serial port.")
            self.stop_reading()

    def closeEvent(self, event):
        try:
            if self.serial_thread is not None:
                self.serial_thread.stop()
            event.accept()
        except Exception as e:
             QMessageBox.critical(self, "Error:",e)

if _name_ == '_main_':
    app = QApplication(sys.argv)
    monitor = SerialMonitor()
    monitor.show()
    sys.exit(app.exec())