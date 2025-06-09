#!/usr/bin/env python

import time
import serial

class PortHandler(object):

    def __init__(self, *, port_name: str, baud_rate: int, rcv_timeout_ms: int):

        is_using: bool
        ser: serial.Serial

        self.is_open = False
        self.baudrate = baud_rate  #DEFAULT_BAUDRATE

        self.packet_start_perf_count_ns: int = 0

        self.packet_timeout_ns :int = 0

        self.tx_time_ns_per_byte :int = 0

        # in ms.
        self.rcv_timeout_ns: int = rcv_timeout_ms * 1_000_000

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
        # self.packet_start_perf_count_ns = self.getCurrentTimeMs()
        self.packet_start_perf_count_ns = time.perf_counter_ns()
        self.packet_timeout_ns = ((self.tx_time_ns_per_byte * packet_length)
                                  + (self.tx_time_ns_per_byte * 3)
                                  + self.rcv_timeout_ns) #  + LATENCY_TIMER

    # def setPacketTimeoutMillis(self, msec):
    #     self.packet_start_perf_count_ns = self.getCurrentTimeMs()
    #     self.packet_timeout_ms = msec

    def checkPacketTimeout(self):
        # print(f'--- check timeout: perf count ms since pkt start: { self.getPerfCntNsSincePktStart() // 1_000_000 } '
        #       f' timeout ms : {self.packet_timeout_ns // 1_000_000}')
        cnt_ns = self.getPerfCntNsSincePktStart()
        if cnt_ns > self.packet_timeout_ns:
            raise IOError(f'rcv pkt timeout--- pls check FT232 USB converter latency_timer --- perf cnt ms since pkt start: {cnt_ns//1_000_000},'
                          f'timeout_ms: { self.packet_timeout_ns // 1_000_000} ')
            # self.packet_timeout_ns = 0
            # return True

        # always clear.
        self.packet_timeout_ns = 0
        return False

    # def getCurrentTimeMs(self)->int:
        # return round(time.time() * 1000000000) / 1000000.0
        # return time.time_ns() // 1000000

    def getPerfCntNsSincePktStart(self):
        # time_since = self.getCurrentTimeMs() - self.packet_start_perf_count_ns
        time_since = time.perf_counter_ns() - self.packet_start_perf_count_ns
        if time_since < 0:
            # self.packet_start_perf_count_ns = self.getCurrentTimeMs()
            raise ValueError(f'packet_start_perf_count_ns: {self.packet_start_perf_count_ns}'
                             f' less than now perf count:{time.perf_counter_ns()}.')
            # self.packet_start_perf_count_ns = time.perf_counter_ns()

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
            timeout=self.rcv_timeout_ns * 2. / 1000.    # 0.2  #1   #0 # in seconds.
        )

        self.is_open = True

        self.ser.reset_input_buffer()

        # for feite actuator bus protocol, 10bit per byte.
        self.tx_time_ns_per_byte: int = round ((1_000_000_000. / self.baudrate) * 10.0)

        return True

    def getCFlagBaud(self, baudrate):
        if baudrate in {4800, 9600, 14400, 19200, 38400, 57600, 115200, 128000, 250000, 500000, 1000000}:
            return baudrate
        else:
            return None