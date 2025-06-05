#!/usr/bin/env python

from .scservo_def import *
from .protocol_packet_handler import *
from .group_sync_read import *
from .group_sync_write import *

SMS_STS_DEFAULT_BAUD_RATE = 115200

#波特率定义
class SMS_STS_Baud_Rate:
    BPS_1M = 0
    BPS_0_5M = 1
    BPS_250K = 2
    BPS_128K = 3
    BPS_115200 = 4
    BPS_76800 = 5
    BPS_57600 = 6
    BPS_38400 = 7

#内存表定义
class SMS_STS_EEPROM_Table_ReadOnly:
    #-------EPROM(只读)--------
    MODEL_L = 3
    # MODEL_H = 4

class SMS_STS_EEPROM_Table_RW:    
    #-------EPROM(读写)--------
    ID = 5
    BAUD_RATE = 6
    RETURN_DELAY_TIME = 7
    MIN_ANGLE_LIMIT_L = 9
    # MIN_ANGLE_LIMIT_H = 10
    MAX_ANGLE_LIMIT_L = 11
    # MAX_ANGLE_LIMIT_H = 12
    KP = 21
    # KD = 22
    # KI = 23
    CW_DEAD = 26
    CCW_DEAD = 27
    OFS_L = 31
    OFS_H = 32
    CONTROL_MODE = 33

class SMS_STS_SRAM_Table_RW:    
    #-------SRAM(读写)--------
    TORQUE_ENABLE = 40
    ACC = 41
    GOAL_POSITION_L = 42
    # GOAL_POSITION_H = 43
    GOAL_TIME_L = 44
    # GOAL_TIME_H = 45
    GOAL_SPEED_L = 46
    # GOAL_SPEED_H = 47
    LOCK = 55

class SMS_STS_SRAM_Table_ReadOnly:
    #-------SRAM(只读)--------
    PRESENT_POSITION_L = 56
    # PRESENT_POSITION_H = 57
    PRESENT_VELOCITY_L = 58
    # PRESENT_VELOCITY_H = 59
    PRESENT_LOAD_L = 60
    # PRESENT_LOAD_H = 61
    PRESENT_VOLTAGE = 62
    PRESENT_TEMPERATURE = 63
    MOVING = 66
    PRESENT_CURRENT_L = 69
    # PRESENT_CURRENT_H = 70

# Register Byte Length
class SMS_STS_Table_Data_Length:
    MODEL_NUMBER = 2
    PRESENT_POSITION = 2
    PRESENT_VELOCITY = 2
    PRESENT_LOAD = 2
    # PRESENT_POS_VEL = 4
    # PRESENT_POS_VEL_LOAD = 6
    GOAL_POSITION = 2
    PRESENT_VOLTAGE = 1
    # LEN_GOAL_CURRENT = 2


class PacketHandler(ProtocolPacketHandler):
    def __init__(self, port_handler):
        super().__init__(self, port_handler=port_handler, byte_order=ByteOrder.LITTLE)
        self.groupSyncWrite = GroupSyncWrite(self, ACC, 7)

    def WritePosEx(self, scs_id, position, velocity, acc):
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), 0, 0, self.scs_lobyte(velocity), self.scs_hibyte(velocity)]
        return self.writeTxRx(scs_id, ACC, len(txpacket), txpacket)

    def ReadPos(self, scs_id):
        scs_present_position, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L)
        return self.scs_tohost(scs_present_position, 15), scs_comm_result, scs_error

    def ReadSpeed(self, scs_id):
        scs_present_velocity, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, SMS_STS_SRAM_Table_ReadOnly.PRESENT_VELOCITY_L)
        return self.scs_tohost(scs_present_velocity, 15), scs_comm_result, scs_error

    def ReadPosSpeed(self, scs_id):
        scs_present_position_velocity, scs_comm_result, scs_error = self.read4ByteTxRx(scs_id, SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L)
        scs_present_position = self.scs_loword(scs_present_position_velocity)
        scs_present_velocity = self.scs_hiword(scs_present_position_velocity)
        return self.scs_tohost(scs_present_position, 15), self.scs_tohost(scs_present_velocity, 15), scs_comm_result, scs_error

    def ReadMoving(self, scs_id):
        moving, scs_comm_result, scs_error = self.read1ByteTxRx(scs_id, SMS_STS_SRAM_Table_ReadOnly.MOVING)
        return moving, scs_comm_result, scs_error

    # NOTE: only add params. when completed, must call packetHandler.groupSyncWrite.txPacket() to send out, then clear params.
    def SyncWritePosEx(self, scs_id, position, velocity, acc):
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), 0, 0, self.scs_lobyte(velocity), self.scs_hibyte(velocity)]
        return self.groupSyncWrite.addParam(scs_id, txpacket)

    def RegWritePosEx(self, scs_id, position, velocity, acc):
        txpacket = [acc, self.scs_lobyte(position), self.scs_hibyte(position), 0, 0, self.scs_lobyte(velocity), self.scs_hibyte(velocity)]
        return self.regWriteTxRx(scs_id, ACC, len(txpacket), txpacket)

    def RegAction(self):
        return self.action(BROADCAST_ID)

    def WheelMode(self, scs_id):
        return self.write1ByteTxRx(scs_id, SMS_STS_EEPROM_Table_RW.MODE, 1)

    def WriteSpec(self, scs_id, velocity, acc):
        velocity = self.scs_toscs(velocity, 15)
        txpacket = [acc, 0, 0, 0, 0, self.scs_lobyte(velocity), self.scs_hibyte(velocity)]
        return self.writeTxRx(scs_id, ACC, len(txpacket), txpacket)

    def LockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, SMS_STS_SRAM_Table_RW.LOCK, 1)

    def unLockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, LOCK, 0)

