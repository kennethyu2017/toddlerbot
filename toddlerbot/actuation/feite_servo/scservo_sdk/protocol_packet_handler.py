#!/usr/bin/env python

from typing import Tuple, List, Optional
from enum import Enum
from .scservo_def import *
from .port_handler import PortHandler

TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 250

# for Protocol Packet
class ProtoPkt:
    HEADER0 = 0
    HEADER1 = 1
    ID = 2
    LENGTH = 3
    INSTRUCTION = 4
    ERROR = 4
    PARAMETER0 = 5

# Protocol Error bit
class ProtoErrBit:
    VOLTAGE = 1
    ANGLE = 2
    OVERHEAT = 4
    OVERELE = 8
    OVERLOAD = 32

# byte order.
# LITTLE: the most significant byte is at the end of the byte array.
class ByteOrder(Enum):
    LITTLE = 1
    BIG = 2

class ProtocolPacketHandler(object):
    def __init__(self, *, port_handler:PortHandler, byte_order:ByteOrder):
        #self.scs_setend(protocol_end)# SCServo bit end(STS/SMS=0, SCS=1)
        self.port_handler = port_handler
        self.byte_order = byte_order

    def scs_getend(self) -> ByteOrder:
        return self.byte_order

    def scs_setend(self, e:ByteOrder):
        self.byte_order = e

    def scs_tohost(self, a, b):
        if (a & (1<<b)):
            return -(a & ~(1<<b))
        else:
            return a

    def scs_toscs(self, a, b):
        if (a<0):
            return (-a | (1<<b))
        else:
            return a

    def scs_makeword(self, a, b):
        if self.byte_order==ByteOrder.LITTLE:            
            return (a & 0xFF) | ((b & 0xFF) << 8)
        else:
            return (b & 0xFF) | ((a & 0xFF) << 8)

    def scs_makedword(self, a, b):
        return (a & 0xFFFF) | (b & 0xFFFF) << 16

    def scs_loword(self, l):
        return l & 0xFFFF

    def scs_hiword(self, h):
        return (h >> 16) & 0xFFFF

    def scs_lobyte(self, w):
        if self.byte_order==ByteOrder.LITTLE:
            return w & 0xFF
        else:
            return (w >> 8) & 0xFF

    def scs_hibyte(self, w):
        if self.byte_order==ByteOrder.LITTLE:
            return (w >> 8) & 0xFF
        else:
            return w & 0xFF
        
    def getProtocolVersion(self):
        return 1.0

    def getTxRxResult(self, result):        
        if result == CommResult.SUCCESS:
            return "[TxRxResult] Communication success!"
        elif result == CommResult.PORT_BUSY:
            return "[TxRxResult] Port is in use!"
        elif result == CommResult.TX_FAIL:
            return "[TxRxResult] Failed transmit instruction packet!"
        elif result == CommResult.RX_FAIL:
            return "[TxRxResult] Failed get status packet from device!"
        elif result == CommResult.TX_ERROR:
            return "[TxRxResult] Incorrect instruction packet!"
        elif result == CommResult.RX_WAITING:
            return "[TxRxResult] Now receiving status packet!"
        elif result == CommResult.RX_TIMEOUT:
            return "[TxRxResult] Rx timeout, there is no status packet!"
        elif result == CommResult.RX_CORRUPT:
            return "[TxRxResult] Incorrect status packet!"
        elif result == CommResult.NOT_AVAILABLE:
            return "[TxRxResult] Protocol does not support this function!"
        else:
            return ""

    def getRxPacketError(self, error:int):
        if error & ProtoErrBit.VOLTAGE:
            return "[ServoStatus] Input voltage error!"

        if error & ProtoErrBit.ANGLE:
            return "[ServoStatus] Angle sen error!"

        if error & ProtoErrBit.OVERHEAT:
            return "[ServoStatus] Overheat error!"

        if error & ProtoErrBit.OVERELE:
            return "[ServoStatus] OverEle error!"
        
        if error & ProtoErrBit.OVERLOAD:
            return "[ServoStatus] Overload error!"

        return ""

    def txPacket(self, txpacket:bytearray)->CommResult:
        checksum = 0
        total_packet_length = txpacket[ProtoPkt.LENGTH] + 4  # 4: HEADER0 HEADER1 ID LENGTH

        if self.port_handler.is_using:
            return CommResult.PORT_BUSY
        self.port_handler.is_using = True

        # check max packet length
        if total_packet_length > TXPACKET_MAX_LEN:
            self.port_handler.is_using = False
            return CommResult.TX_ERROR

        # make packet header
        txpacket[ProtoPkt.HEADER0] = 0xFF
        txpacket[ProtoPkt.HEADER1] = 0xFF

        # add a checksum to the packet
        for idx in range(2, total_packet_length - 1):  # except header, checksum
            checksum += txpacket[idx]

        txpacket[total_packet_length - 1] = ~checksum & 0xFF

        #print "[TxPacket] %r" % txpacket

        # tx packet
        self.port_handler.clearPort()
        written_packet_length = self.port_handler.writePort(txpacket)
        if total_packet_length != written_packet_length:
            self.port_handler.is_using = False
            return CommResult.TX_FAIL

        return CommResult.SUCCESS

    def rxPacket(self)->Tuple[bytearray, CommResult]:
        rxpacket = bytearray()  # mutable.

        result = CommResult.TX_FAIL
        checksum = 0
        rx_length = 0
        wait_length = 6  # minimum length (HEADER0 HEADER1 ID LENGTH ERROR CHKSUM)

        while True:
            rxpacket.extend(self.port_handler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                # find packet header
                for idx in range(0, (rx_length - 1)):
                    if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
                        break

                if idx == 0:  # found at the beginning of the packet
                    if (rxpacket[ProtoPkt.ID] > 0xFD) or (rxpacket[ProtoPkt.LENGTH] > RXPACKET_MAX_LEN) or (
                            rxpacket[ProtoPkt.ERROR] > 0x7F):
                        # unavailable ID or unavailable Length or unavailable Error
                        # remove the first byte in the packet
                        del rxpacket[0]
                        rx_length -= 1
                        continue

                    # re-calculate the exact length of the rx packet
                    if wait_length != (rxpacket[ProtoPkt.LENGTH] + ProtoPkt.LENGTH + 1):
                        wait_length = rxpacket[ProtoPkt.LENGTH] + ProtoPkt.LENGTH + 1
                        continue

                    if rx_length < wait_length:
                        # check timeout
                        if self.port_handler.isPacketTimeout():
                            if rx_length == 0:
                                result = CommResult.RX_TIMEOUT
                            else:
                                result = CommResult.RX_CORRUPT
                            break
                        else:
                            continue

                    # calculate checksum
                    for i in range(2, wait_length - 1):  # except header, checksum
                        checksum += rxpacket[i]
                    checksum = ~checksum & 0xFF

                    # verify checksum
                    if rxpacket[wait_length - 1] == checksum:
                        result = CommResult.SUCCESS
                    else:
                        result = CommResult.RX_CORRUPT
                    break

                else:
                    # remove unnecessary packets
                    del rxpacket[0: idx]
                    rx_length -= idx

            else:
                # check timeout
                if self.port_handler.isPacketTimeout():
                    if rx_length == 0:
                        result = CommResult.RX_TIMEOUT
                    else:
                        result = CommResult.RX_CORRUPT
                    break

        self.port_handler.is_using = False
        return rxpacket, result

    def txRxPacket(self, txpacket:bytearray) -> Tuple[Optional[bytearray], CommResult, ProtoErrBit]:
        rxpacket = None
        error:int = 0
        result = CommResult.TX_FAIL

        # tx packet
        result = self.txPacket(txpacket)
        if result != CommResult.SUCCESS:
            return None, result, error

        # (ID == Broadcast ID) == no need to wait for status packet or not available
        if (txpacket[ProtoPkt.ID] == BROADCAST_ID):
            self.port_handler.is_using = False
            return None, result, error

        # set packet timeout
        if txpacket[ProtoPkt.INSTRUCTION] == SCSProtoInst.READ:
            self.port_handler.setPacketTimeout(txpacket[ProtoPkt.PARAMETER0 + 1] + 6)
        else:
            self.port_handler.setPacketTimeout(6)  # HEADER0 HEADER1 ID LENGTH ERROR CHECKSUM

        # rx packet
        while True:
            _pkt, result = self.rxPacket()
            if result != CommResult.SUCCESS:
                break
            elif _pkt[ProtoPkt.ID] == txpacket[ProtoPkt.ID]:
                rxpacket = _pkt
                break

        if result == CommResult.SUCCESS:
            assert txpacket[ProtoPkt.ID] == rxpacket[ProtoPkt.ID]
            error = rxpacket[ProtoPkt.ERROR]
            return rxpacket, result, error

        return None, result, error

    def ping(self, scs_id:int)->Tuple[int,CommResult,int]:
        model_number = 0
        error = 0

        txpacket = bytearray(6)  # 6-bytes for ping pkt.

        if scs_id > BROADCAST_ID:
            return model_number, CommResult.NOT_AVAILABLE, error

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = 2
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.PING

        rxpacket, result, error = self.txRxPacket(txpacket)

        if result == CommResult.SUCCESS:
            data_read, result, error = self.readTxRx(scs_id, 3, 2)  # Address 3 : Model Number
            if result == CommResult.SUCCESS:
                model_number = self.scs_makeword(data_read[0], data_read[1])

        return model_number, result, error

    def action(self, scs_id)->CommResult:
        txpacket = bytearray(6)   #  6-bytes action pkt.

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = 2
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.ACTION

        _, result, _ = self.txRxPacket(txpacket)

        return result

    def readTx(self, scs_id, address, length)->CommResult:

        txpacket = bytearray(8)

        if scs_id >= BROADCAST_ID:
            return CommResult.NOT_AVAILABLE

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = 4
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.READ
        txpacket[ProtoPkt.PARAMETER0 + 0] = address
        txpacket[ProtoPkt.PARAMETER0 + 1] = length

        result = self.txPacket(txpacket)

        # set packet timeout
        if result == CommResult.SUCCESS:
            self.port_handler.setPacketTimeout(length + 6)

        return result

    def readRx(self, scs_id:int, length:int)->Tuple[Optional[bytearray], CommResult, int]:
        result = CommResult.TX_FAIL
        error = 0

        rxpacket = None

        while True:
            _pkt, result = self.rxPacket()

            if result != CommResult.SUCCESS:
                break
            elif _pkt[ProtoPkt.ID] == scs_id:
                rxpacket=_pkt
                break

        if result == CommResult.SUCCESS:
            assert len(rxpacket) > ProtoPkt.PARAMETER0 + length and rxpacket[ProtoPkt.ID] == scs_id
            error = rxpacket[ProtoPkt.ERROR]
            return rxpacket[ProtoPkt.PARAMETER0 : ProtoPkt.PARAMETER0+length], result, error

        return None, result, error

    def readTxRx(self, scs_id, address, length)->Tuple[Optional[bytearray], CommResult, int]:

        txpacket = bytearray(8)

        if scs_id >= BROADCAST_ID:
            return None, CommResult.NOT_AVAILABLE, 0

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = 4
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.READ
        txpacket[ProtoPkt.PARAMETER0 + 0] = address
        txpacket[ProtoPkt.PARAMETER0 + 1] = length

        rxpacket, result, error = self.txRxPacket(txpacket)
        if result == CommResult.SUCCESS:
            assert len(rxpacket) > ProtoPkt.PARAMETER0 + length and rxpacket[ProtoPkt.ID] == scs_id
            error = rxpacket[ProtoPkt.ERROR]
            return rxpacket[ProtoPkt.PARAMETER0 : ProtoPkt.PARAMETER0+length], result,error

        return None, result, error

    def read1ByteTx(self, scs_id, address)->CommResult:
        return self.readTx(scs_id, address, 1)

    def read1ByteRx(self, scs_id)->Tuple[int, CommResult, int]:
        data, result, error = self.readRx(scs_id, 1)
        data_read = data[0] if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def read1ByteTxRx(self, scs_id, address)->Tuple[int, CommResult, int]:
        data, result, error = self.readTxRx(scs_id, address, 1)
        data_read = data[0] if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def read2ByteTx(self, scs_id, address)->CommResult:
        return self.readTx(scs_id, address, 2)

    def read2ByteRx(self, scs_id)->Tuple[int, CommResult, int]:
        data, result, error = self.readRx(scs_id, 2)
        data_read = self.scs_makeword(data[0], data[1]) if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def read2ByteTxRx(self, scs_id, address)->Tuple[int, CommResult, int]:
        data, result, error = self.readTxRx(scs_id, address, 2)
        data_read = self.scs_makeword(data[0], data[1]) if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def read4ByteTx(self, scs_id, address)->CommResult:
        return self.readTx(scs_id, address, 4)

    def read4ByteRx(self, scs_id)->Tuple[int, CommResult, int]:
        data, result, error = self.readRx(scs_id, 4)
        data_read = self.scs_makedword(self.scs_makeword(data[0], data[1]),
                                  self.scs_makeword(data[2], data[3])) if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def read4ByteTxRx(self, scs_id, address)->Tuple[int, CommResult, int]:
        data, result, error = self.readTxRx(scs_id, address, 4)
        data_read = self.scs_makedword(self.scs_makeword(data[0], data[1]),
                                  self.scs_makeword(data[2], data[3])) if (result == CommResult.SUCCESS) else 0
        return data_read, result, error

    def writeTxOnly(self, scs_id, address, length, data)->CommResult:
        txpacket = bytearray(length + 7)

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = length + 3
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.WRITE
        txpacket[ProtoPkt.PARAMETER0] = address

        txpacket[ProtoPkt.PARAMETER0 + 1: ProtoPkt.PARAMETER0 + 1 + length] = data[0: length]

        result = self.txPacket(txpacket)
        self.port_handler.is_using = False

        return result

    # TODO: get length from len(data).remove `length` arg.
    def writeTxRx(self, *,
                  scs_id, address, length, data:bytes|bytearray)->Tuple[CommResult, int]:
        txpacket = bytearray(length + 7)

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = length + 3
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.WRITE
        txpacket[ProtoPkt.PARAMETER0] = address

        txpacket[ProtoPkt.PARAMETER0 + 1: ProtoPkt.PARAMETER0 + 1 + length] = data[0: length]
        rxpacket, result, error = self.txRxPacket(txpacket)

        return result, error

    def write1ByteTxOnly(self, scs_id, address, data:bytes| bytearray)->CommResult:
        assert len(data) == 1
        return self.writeTxOnly(scs_id, address, 1, data)

    def write1ByteTxRx(self, scs_id, address, data:bytes| bytearray)->Tuple[CommResult,int]:
        assert len(data) == 1
        return self.writeTxRx(scs_id, address, 1, data)

    def write2ByteTxOnly(self, scs_id, address, data:bytes|bytearray)->CommResult:
        assert len(data) == 2
        return self.writeTxOnly(scs_id, address, 2, data)

    def write2ByteTxRx(self, scs_id, address, data:bytes|bytearray)->Tuple[CommResult, int]:
        assert len(data) == 2
        return self.writeTxRx(scs_id, address, 2, data)

    def write4ByteTxOnly(self, scs_id, address, data:bytes|bytearray) -> CommResult:
        # data_write = [self.scs_lobyte(self.scs_loword(data)),
        #               self.scs_hibyte(self.scs_loword(data)),
        #               self.scs_lobyte(self.scs_hiword(data)),
        #               self.scs_hibyte(self.scs_hiword(data))]
        assert len(data) == 4
        return self.writeTxOnly(scs_id, address, 4, data)

    def write4ByteTxRx(self, scs_id, address, data:bytes|bytearray)->Tuple[CommResult,int]:
        # data_write = [self.scs_lobyte(self.scs_loword(data)),
        #               self.scs_hibyte(self.scs_loword(data)),
        #               self.scs_lobyte(self.scs_hiword(data)),
        #               self.scs_hibyte(self.scs_hiword(data))]
        assert len(data)==4
        return self.writeTxRx(scs_id, address, 4, data)

    def regWriteTxOnly(self, scs_id, address, length, data:bytes|bytearray)->CommResult:
        txpacket = bytearray(length + 7)

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = length + 3
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.REG_WRITE
        txpacket[ProtoPkt.PARAMETER0] = address

        txpacket[ProtoPkt.PARAMETER0 + 1: ProtoPkt.PARAMETER0 + 1 + length] = data[0: length]

        result = self.txPacket(txpacket)
        self.port_handler.is_using = False

        return result

    def regWriteTxRx(self, scs_id, address, length, data:bytes|bytearray)->Tuple[CommResult,int]:
        txpacket = bytearray(length + 7)

        txpacket[ProtoPkt.ID] = scs_id
        txpacket[ProtoPkt.LENGTH] = length + 3
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.REG_WRITE
        txpacket[ProtoPkt.PARAMETER0] = address

        txpacket[ProtoPkt.PARAMETER0 + 1: ProtoPkt.PARAMETER0 + 1 + length] = data[0: length]

        _, result, error = self.txRxPacket(txpacket)

        return result, error

    def syncReadTx(self, start_address, data_length, param:bytes|bytearray, param_length)->CommResult:
        txpacket = bytearray(param_length + 8)
        # 8: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN CHKSUM

        txpacket[ProtoPkt.ID] = BROADCAST_ID
        txpacket[ProtoPkt.LENGTH] = param_length + 4  # 7: INST START_ADDR DATA_LEN CHKSUM
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.SYNC_READ
        txpacket[ProtoPkt.PARAMETER0 + 0] = start_address
        txpacket[ProtoPkt.PARAMETER0 + 1] = data_length

        txpacket[ProtoPkt.PARAMETER0 + 2: ProtoPkt.PARAMETER0 + 2 + param_length] = param[0: param_length]

        # print(txpacket)
        result = self.txPacket(txpacket)
        return result

    def syncReadRx(self, data_length, param_length)->Tuple[CommResult,bytearray]:
        wait_length = (6 + data_length) * param_length
        self.port_handler.setPacketTimeout(wait_length)

        rxpacket = bytearray()

        rx_length = 0
        while True:
            rxpacket.extend(self.port_handler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                result = CommResult.SUCCESS
                break
            else:
                # check timeout
                if self.port_handler.isPacketTimeout():
                    if rx_length == 0:
                        result = CommResult.RX_TIMEOUT
                    else:
                        result = CommResult.RX_CORRUPT
                    break
        self.port_handler.is_using = False
        return result, rxpacket
    
    def syncWriteTxOnly(self, start_address, data_length, param:bytes|bytearray, param_length) -> CommResult:
        txpacket = bytearray(param_length + 8)
        # 8: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN ... CHKSUM

        txpacket[ProtoPkt.ID] = BROADCAST_ID
        txpacket[ProtoPkt.LENGTH] = param_length + 4  # 4: INST START_ADDR DATA_LEN ... CHKSUM
        txpacket[ProtoPkt.INSTRUCTION] = SCSProtoInst.SYNC_WRITE
        txpacket[ProtoPkt.PARAMETER0 + 0] = start_address
        txpacket[ProtoPkt.PARAMETER0 + 1] = data_length

        txpacket[ProtoPkt.PARAMETER0 + 2: ProtoPkt.PARAMETER0 + 2 + param_length] = param[0: param_length]

        _, result, _ = self.txRxPacket(txpacket)

        return result
