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
                                                          CommResult, parse_pos, parse_vel)

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_START_ID : int = 0
MOTOR_END_ID: int = 1

def _sync_read_pos_vel_helper(*, reader:GroupSyncReader, id_range:range,):

    # clear param before read.
    reader.clearParam()

    for _id in id_range:
        # Add parameter storage for SCServo#1~10 present position value
        add_param_result = reader.addParam(_id)
        if add_param_result != True:
            print(f"[ID: {_id:03d}] sync reader add param failed")

    comm_result = reader.txRxPacket()
    if comm_result != CommResult.SUCCESS:
        # print(packet_handler.getTxRxResult(comm_result))
        raise ValueError(f'sync reader.txRxPacket return is not success: {comm_result} ')

    errored_ids: List[int] = []
    for _id in id_range:
        # Check if group sync read data of SCServo#1~10 is available
        if not reader.isAvailable(_id,
                                  SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                  4):
            errored_ids.append(_id)
            raise ValueError(f"[ID:{_id:03d} ] groupSyncRead getdata failed")

        else:
            # Get SCServo#scs_id present position value
            present_pos = reader.getDataAsBytes(scs_id=_id,
                                                     address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                                     size=2)

            present_vel = reader.getDataAsBytes(scs_id=_id,
                                                     address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VELOCITY_L,
                                                     size=2)

            print(f'Read [ID: {_id:>3d}] ---> '
                  f' Present Pos:{parse_pos(present_pos): 4.4f}'
                  f' Present Vel:{parse_vel(present_vel): 4.4f}')


def _main():

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    port_handler = PortHandler(port_name=URT_1_DEV_NAME,
                               baud_rate=SMS_STS_DEFAULT_BAUD_RATE,
                               rcv_timeout_ms=10)  # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    packet_handler = ProtocolPacketHandler(port_handler=port_handler,
                                           byte_order=ByteOrder.LITTLE)

    # Open port
    if port_handler.openPort():
        print("Succeeded to open the port")
    else:
        raise IOError("Failed to open the port")

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

            _sync_read_pos_vel_helper(reader=sync_reader,id_range=range(MOTOR_START_ID, MOTOR_END_ID))

            # if scs_error != 0:
            #     print(packet_handler.getRxPacketError(scs_error))
            # sync_reader.clearParam()

    except KeyboardInterrupt:
        print(f'keyboard interrupt, closing port and exit...')

    finally:
        # Close port
        print(f'closing serial port...')
        port_handler.closePort()


if __name__  == '__main__':
    _main()