#!/usr/bin/env python

import time
import serial

class PortHandler(object):

    def __init__(self, *, port_name: str, baud_rate: int, rcv_timeout_ms: int):

        is_using: bool
        ser: serial.Serial

        self.is_open = False
        self.baudrate = baud_rate  #DEFAULT_BAUDRATE
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0

        # in ms.
        self.rcv_timeout_ms = rcv_timeout_ms

        self.is_using = False
        self.port_name = port_name
        self.ser = None

    def openPort(self):
        return self.setBaudRate(self.baudrate)

    def closePort(self):
        self.ser.close()
        self.is_open = False

    def clearPort(self):
        self.ser.flush()

    def setPortName(self, port_name):
        self.port_name = port_name

    def getPortName(self):
        return self.port_name

    # NOTE: this method just set RS485 serial baud rate, not the value in EEPROM table of motors.
    def setBaudRate(self, baudrate):
        baud = self.getCFlagBaud(baudrate)

        if baud is None:
            # self.setupPort(38400)
            # self.baudrate = baudrate
            return False  # TODO: setCustomBaudrate(baudrate)
        else:
            self.baudrate = baudrate
            # TODO: set motor EEPROM table same time?
            return self.setupPort(baud)

    def getBaudRate(self):
        return self.baudrate

    def getBytesAvailable(self):
        return self.ser.in_waiting

    def readPort(self, length)-> bytes:
        # if (sys.version_info > (3, 0)):
        return self.ser.read(length)
        # else:
        #     return [ord(ch) for ch in self.ser.read(length)]

    def writePort(self, packet:[bytes | bytearray]):
        return self.ser.write(packet)

    def setPacketTimeout(self, packet_length):
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + self.rcv_timeout_ms #  + LATENCY_TIMER

    def setPacketTimeoutMillis(self, msec):
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = msec

    def isPacketTimeout(self):
        if self.getTimeSinceStart() > self.packet_timeout:
            self.packet_timeout = 0
            return True

        return False

    def getCurrentTime(self):
        return round(time.time() * 1000000000) / 1000000.0

    def getTimeSinceStart(self):
        time_since = self.getCurrentTime() - self.packet_start_time
        if time_since < 0.0:
            self.packet_start_time = self.getCurrentTime()

        return time_since

    def setupPort(self, cflag_baud):
        if self.is_open:
            self.closePort()

        # Feite SM40BL using 8N1
        self.ser = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            parity = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            # NOTE: in seconds. timeout will decide the read() behaviour, see
            # pySerial doc.
            # if timeout is None (default arg value), it will block until the requested number of bytes is read.
            # TODO: we use the 2 times timeout_ms to give more tolerance for serial port, and make upper layer
            # code to choose.
            # NOTE: in seconds.
            timeout=self.rcv_timeout_ms * 2. / 1000.    # 0.2  #1   #0 # in seconds.
        )

        self.is_open = True

        self.ser.reset_input_buffer()

        # in ms.
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0

        return True

    def getCFlagBaud(self, baudrate):
        if baudrate in {4800, 9600, 14400, 19200, 38400, 57600, 115200, 128000, 250000, 500000, 1000000}:
            return baudrate
        else:
            return None