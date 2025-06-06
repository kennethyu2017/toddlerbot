#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#
from typing import List
import time
from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          GroupSyncReader,SMS_STS_SRAM_Table_ReadOnly,
                                                          CommResult)                   # Uses FTServo SDK library
# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'


if __name__  == '__main__':

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    port_handler = PortHandler(port_name=URT_1_DEV_NAME,
                               baud_rate=SMS_STS_DEFAULT_BAUD_RATE,
                               latency_timer=10)  # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    packet_handler = ProtocolPacketHandler(port_handler=port_handler,
                                           byte_order=ByteOrder.LITTLE)

    # Open port
    if port_handler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        quit()
    try:
        while True:
            time.sleep(1.)

            # Read the current position of servo motor (ID1)
            scs_present_position, scs_present_speed, scs_comm_result, scs_error = packet_handler.(1)
            if scs_comm_result != CommResult.SUCCESS:
                print(packet_handler.getTxRxResult(scs_comm_result))
            else:
                print("[ID:%03d] PresPos:%d PresSpd:%d" % (1, scs_present_position, scs_present_speed))
            if scs_error != 0:
                print(packet_handler.getRxPacketError(scs_error))
            time.sleep(1)


    finally:
        # Close port
        portHandler.closePort()
