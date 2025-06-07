#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#
import time
from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          read_pos_and_vel_helper
                                                          )

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_ID :int = 0


if __name__  == '__main__':

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    _port_handler = PortHandler(port_name=URT_1_DEV_NAME,
                               baud_rate=SMS_STS_DEFAULT_BAUD_RATE,
                               rcv_timeout_ms=10)  # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    _packet_handler = ProtocolPacketHandler(port_handler=_port_handler,
                                           byte_order=ByteOrder.LITTLE)

    # Open port
    if _port_handler.openPort():
        print("Succeeded to open the port")
    else:
        raise IOError("Failed to open the port")

    try:
        while True:
            time.sleep(1.)
            read_pos_and_vel_helper(_packet_handler, MOTOR_ID)

    finally:
        # Close port
        _port_handler.closePort()
