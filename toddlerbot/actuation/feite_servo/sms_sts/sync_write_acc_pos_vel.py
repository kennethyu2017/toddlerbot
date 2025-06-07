#!/usr/bin/env python
#
# *********     Sync Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS

import time
from typing import List
from numpy import pi

from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder, GroupSyncReader,
                                                         GroupSyncWriter, construct_acc_pos_vel_txpkt_helper,
                                                          SMS_STS_SRAM_Table_RW, SMS_STS_SRAM_Table_ReadOnly,
                                                          CommResult)

from toddlerbot.actuation.feite_servo.sms_sts.sync_read_pos_vel import _sync_read_pos_vel_helper

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_START_ID : int = 0
MOTOR_END_ID : int = 1

def _sync_write_acc_pos_vel_helper(*, writer: GroupSyncWriter,
                                   id_range: range,
                                   acc_radius:float,
                                   pos_radius:float,
                                   vel_radius:float):
    # comm_result, error = packet_handler.WritePosEx(1, 4095, 60, 50)
    txpkt: bytes = construct_acc_pos_vel_txpkt_helper(acc_radius=acc_radius,
                                                      pos_radius=pos_radius,
                                                      vel_radius=vel_radius)
    # Clear before addParam.
    writer.clearParam()

    errored_ids: List[int] = []
    for _id in id_range:
        # add same value for all ID.
        success = writer.addParam(_id, txpkt)
        if not success:
            errored_ids.append(_id)

    if errored_ids:
        raise ValueError(f"Sync write failed for: {errored_ids}")

    comm_result = writer.txPacket()
    if comm_result != CommResult.SUCCESS:
        raise ValueError(f'sync_writer.txPacket result is not success: {comm_result} ')


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
        print("Succeed to open the port")
    else:
        raise IOError("Failed to open the port")

    sync_writer = GroupSyncWriter(
        packet_handler=packet_handler,
        start_address=SMS_STS_SRAM_Table_RW.ACC,
        data_length=7 )

    sync_reader = GroupSyncReader(packet_handler=packet_handler,
                                  start_address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                  data_length=4)

    try:
        while True:

            time.sleep(0.5)

            # Servo (ID1) runs at a maximum speed of V=60 * 0.732=43.92rpm
            # and an acceleration of A=50 * 8.7deg/s ^ 2 until it reaches position P1=4095
            TARGET_ACC_RADIUS: float = 5.0  # 7.5
            TARGET_VEL_RADIUS: float = 4.5
            TARGET_POS_RADIUS: float = 2 * pi * .99

            _sync_write_acc_pos_vel_helper(writer=sync_writer,
                                           id_range=range(MOTOR_START_ID, MOTOR_END_ID),
                                           acc_radius=TARGET_ACC_RADIUS,
                                           vel_radius=TARGET_VEL_RADIUS,
                                           pos_radius=TARGET_POS_RADIUS)


            time.sleep(((2 * pi - 0) / TARGET_VEL_RADIUS + TARGET_VEL_RADIUS / TARGET_ACC_RADIUS + 0.1))  # [(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05

            _sync_read_pos_vel_helper(reader=sync_reader, id_range=range(MOTOR_START_ID, MOTOR_END_ID) )

            # Servo (ID1) runs at a maximum speed of V=60 * 0.732=43.92rpm and an acceleration of A=50 * 8.7deg/s ^ 2 until P0=0 position
            # scs_comm_result, scs_error = packet_handler.WritePosEx(1, 0, 60, 50)
            TARGET_POS_RADIUS = 0.

            _sync_write_acc_pos_vel_helper(writer=sync_writer,
                                           id_range=range(MOTOR_START_ID, MOTOR_END_ID),
                                           acc_radius=TARGET_ACC_RADIUS,
                                           vel_radius=TARGET_VEL_RADIUS,
                                           pos_radius=TARGET_POS_RADIUS)

            time.sleep( (( 2 * pi - 0) / TARGET_VEL_RADIUS + TARGET_VEL_RADIUS / TARGET_ACC_RADIUS + 0.1))  # [(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05

            _sync_read_pos_vel_helper(reader=sync_reader, id_range=range(MOTOR_START_ID, MOTOR_END_ID))

    finally:
        # Close port
        port_handler.closePort()


if __name__  == '__main__':
    _main()