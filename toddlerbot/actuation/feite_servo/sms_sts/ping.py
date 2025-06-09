#!/usr/bin/env python
#
# *********     Ping Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#

import time
from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          CommResult)                   # Uses FTServo SDK library
# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'

def _main():
    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    port_handler = PortHandler(port_name=URT_1_DEV_NAME,
                               baud_rate=SMS_STS_DEFAULT_BAUD_RATE,
                               rcv_timeout_ms=10) #ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    packet_handler = ProtocolPacketHandler(port_handler=port_handler,
                                           byte_order=ByteOrder.LITTLE)
    # Open port
    if port_handler.openPort():
        print("Succeeded to open the port")
    else:
        raise IOError("Failed to open the port")

        # Set port baudrate 1000000
    # if port_handler.setBaudRate(1000000):
    #     print("Succeeded to change the baudrate")
    # else:
    #     print("Failed to change the baudrate")
    #     quit()

    # Try to ping the ID:1 FTServo
    # Get SCServo model number
    for _ in range(100):
        scs_model_number, scs_comm_result, scs_error = packet_handler.ping(scs_id=0)
        if scs_comm_result != CommResult.SUCCESS:
            print("%s" % packet_handler.getTxRxResult(scs_comm_result))
        else:
            print("[ID:%03d] ping Succeeded. SCServo model number : %d" % (1, scs_model_number))
        if scs_error != 0:
            print("%s" % packet_handler.getRxPacketError(scs_error))

        time.sleep(1.)

    # Close port
    port_handler.closePort()


if __name__ == '__main__':
    _main()

