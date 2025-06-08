#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#

import time
from numpy import pi

from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler, SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          read_pos_and_vel_helper, write_acc_pos_vel_helper,
                                                          )

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_ID :int = 0


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

    try:
        while True:
            time.sleep(.5)
            # Servo (ID1) runs at a maximum speed of V=60 * 0.732=43.92rpm
            # and an acceleration of A=50 * 8.7deg/s ^ 2 until it reaches position P1=4095
            TARGET_ACC_RADIUS: float = 5.0  # 7.5
            TARGET_VEL_RADIUS: float = 4.5
            TARGET_POS_RADIUS: float = 2 * pi * .99

            # comm_result, error = packet_handler.WritePosEx(1, 4095, 60, 50)
            write_acc_pos_vel_helper(writer=packet_handler,
                                      motor_id=MOTOR_ID,
                                      acc_radius=TARGET_ACC_RADIUS,
                                      pos_radius=TARGET_POS_RADIUS,
                                      vel_radius=TARGET_VEL_RADIUS
                                      )

            # time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))#[(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05
            time.sleep((( 2 * pi - 0) / TARGET_VEL_RADIUS + TARGET_VEL_RADIUS / TARGET_ACC_RADIUS + 0.1))  # [(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05

            read_pos_and_vel_helper(packet_handler, MOTOR_ID)

            # Servo (ID1) runs at a maximum speed of V=60 * 0.732=43.92rpm and an acceleration of A=50 * 8.7deg/s ^ 2 until P0=0 position
            # scs_comm_result, scs_error = packet_handler.WritePosEx(1, 0, 60, 50)

            TARGET_POS_RADIUS = 0.
            write_acc_pos_vel_helper(writer=packet_handler,
                                      motor_id=MOTOR_ID,
                                      acc_radius=TARGET_ACC_RADIUS,
                                      pos_radius=TARGET_POS_RADIUS,
                                      vel_radius=TARGET_VEL_RADIUS
                                      )
            # time.sleep(((4095-0)/(60*50) + (60*50)/(50*100) + 0.05))#[(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05
            time.sleep((( 2 * pi - 0) / TARGET_VEL_RADIUS + TARGET_VEL_RADIUS / TARGET_ACC_RADIUS + 0.1))  # [(P1-P0)/(V*50)] + [(V*50)/(A*100)] + 0.05

            read_pos_and_vel_helper(packet_handler, MOTOR_ID)

    finally:
        # Close port
        port_handler.closePort()


if __name__  == '__main__':
    _main()
