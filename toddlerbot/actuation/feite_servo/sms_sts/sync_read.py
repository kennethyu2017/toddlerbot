#!/usr/bin/env python
#
# *********     Sync Read Example      *********
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
                                                          CommResult)

from toddlerbot.actuation.feite_client import _parse_pos, _parse_vel

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


    # # Set port baudrate 1000000
    # if portHandler.setBaudRate(1000000):
    #     print("Succeeded to change the baudrate")
    # else:
    #     print("Failed to change the baudrate")
    #     quit()
    #

    sync_reader = GroupSyncReader(packet_handler=packet_handler,
                                  start_address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                  data_length=4)
    try:
        while True:

            time.sleep(1)

            for scs_id in range(0, 1):
                # Add parameter storage for SCServo#1~10 present position value
                scs_addparam_result = sync_reader.addParam(scs_id)
                if scs_addparam_result != True:
                    print(f"[ID: {scs_id:03d}] groupSyncRead addparam failed")

            scs_comm_result = sync_reader.txRxPacket()
            if scs_comm_result != CommResult.SUCCESS:
                print( packet_handler.getTxRxResult(scs_comm_result))

            errored_ids: List[int] = []
            for _id in range(0, 1):
                # Check if groupsyncread data of SCServo#1~10 is available
                if not sync_reader.isAvailable(_id,
                                               SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                               4):
                    errored_ids.append(_id)
                    print(f"[ID:{_id:03d} ] groupSyncRead getdata failed")
                    continue

                else:
                    # Get SCServo#scs_id present position value
                    present_pos = sync_reader.getDataAsBytes(scs_id=_id,
                                                                      address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                                                      size=2)

                    present_vel = sync_reader.getDataAsBytes(scs_id=_id,
                                                                   address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VELOCITY_L,
                                                                   size=2)

                    print(f' [ID: {_id:>3d}] '
                          f' Present Pos:{ _parse_pos(present_pos): 4.4f}'
                          f' Present Vel:{ _parse_vel(present_vel): 4.4f}' )


                # if scs_error != 0:
                #     print(packet_handler.getRxPacketError(scs_error))

            sync_reader.clearParam()

    except KeyboardInterrupt:
        print(f'keyboard interrupt, closing port and exit...')

    finally:
        # Close port
        print(f'closing serial port...')
        port_handler.closePort()


