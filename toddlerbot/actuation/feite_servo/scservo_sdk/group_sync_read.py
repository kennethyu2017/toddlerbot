#!/usr/bin/env python

from typing import List, Dict,Optional, Tuple, NamedTuple, NoReturn
from .scservo_def import *
from .protocol_packet_handler import ProtocolPacketHandler

class RetData(NamedTuple):
    error: int
    params: bytearray

class GroupSyncReader:
    def __init__(self, *, packet_handler:ProtocolPacketHandler, 
                 start_address:int, data_length:int):
        param:bytearray

        # --- NOTE: first byte of self.data_dict[scs_id] is error code, after which are readed params.---
        _recv_data_dict: Dict[int, RetData]

        self.packet_handler = packet_handler
        self.start_address = start_address
        self.data_length = data_length

        self.last_result = False
        self.is_param_changed = False

        # NOTE: _recv_data_dict,param are set/cleared after every sync_read.
        self.param = bytearray()
        # key is motor id.
        self._rcv_data_dict = {}
       

    def makeParam(self)->bool:
        if len(self._rcv_data_dict) == 0:
            return False

        self.param.clear()

        for scs_id in self._rcv_data_dict:
            self.param.append(scs_id)
            
        return True

    # NOTE: _recv_data_dict,param are set/cleared at every sync_read.
    def addParam(self, scs_id:int)->bool:
        if scs_id in self._rcv_data_dict:  # scs_id already exist
            return False

        self._rcv_data_dict[scs_id] = None # RetData(0, bytearray()) # (error_code, params)  # [0] * self.data_length

        self.is_param_changed = True
        return True

    def removeParam(self, scs_id):
        if scs_id not in self._rcv_data_dict:  # NOT exist
            return

        del self._rcv_data_dict[scs_id]

        self.is_param_changed = True

    #NOTE: _recv_data_dict,param are set/cleared at every sync_read.        
    def clearParam(self):
        self._rcv_data_dict.clear()
        self.param.clear()

    def txPacket(self):
        if len(self._rcv_data_dict.keys()) == 0:
            return CommResult.NOT_AVAILABLE

        if self.is_param_changed is True or len(self.param)==0:            
            if not self.makeParam():
                assert False
                return CommResult.TX_ERROR            

        # print(f'txParam --->')
        # for _b in self.param:
        #     print(f'0x{_b:02x}')

        return self.packet_handler.syncReadTx(self.start_address, self.data_length, self.param, len(self._rcv_data_dict.keys()))

    def rxPacket(self)->CommResult:
        rxpacket:bytearray

        self.last_result = True
        result = CommResult.RX_FAIL

        if len(self._rcv_data_dict.keys()) == 0:
            return CommResult.NOT_AVAILABLE

        result, rxpacket = self.packet_handler.syncReadRx(self.data_length, len(self._rcv_data_dict.keys()))

        # print(f'sync read rx: {rxpacket}')

        if len(rxpacket) >= (self.data_length+6):
            for scs_id in self._rcv_data_dict:
                self._rcv_data_dict[scs_id], result = self.readRx(rxpacket, scs_id, self.data_length)
                if result != CommResult.SUCCESS:
                    self.last_result = False
                # print(scs_id)
        else:
            self.last_result = False

        # print(self.last_result)
        return result

    def txRxPacket(self)->CommResult:
        result = self.txPacket()
        if result != CommResult.SUCCESS:
            return result

        return self.rxPacket()

    # TODO: parsing the rxpacket instead search for each scs_id.
    def readRx(self, rxpacket:bytearray, scs_id, data_length)->Tuple[Optional[RetData], CommResult]:
        # print(scs_id)
        # print(rxpacket)
        ret_data = None
        rx_length = len(rxpacket)
        # print(rx_length)
        rx_index = 0;
        # TODO: optimize, not search whole rxpacket for every scs_id. 
        while (rx_index+6+data_length) <= rx_length:
            headpacket = [0x00, 0x00, 0x00]
            while rx_index < rx_length:
                headpacket[2] = headpacket[1];
                headpacket[1] = headpacket[0];
                headpacket[0] = rxpacket[rx_index];
                rx_index += 1
                if (headpacket[2] == 0xFF) and (headpacket[1] == 0xFF) and headpacket[0] == scs_id:
                    # print(rx_index)
                    break
            # print(rx_index+3+data_length)
            if (rx_index+3+data_length) > rx_length:
                break;
            if rxpacket[rx_index] != (data_length+2):
                rx_index += 1
                # print(rx_index)
                continue
            rx_index += 1
            Error = rxpacket[rx_index]
            rx_index += 1  #point to start of params.
            # NOTE: make a copy to avoid the buffer of data will be modified after readRx.
            ret_params: bytearray = rxpacket[rx_index: rx_index + data_length].copy()
            calSum = scs_id + (data_length+2) + Error
            # Data includes error code.
            # data = [Error]
            # data.extend(rxpacket[rx_index : rx_index+data_length])

            for i in range(0, data_length):
                calSum += rxpacket[rx_index]
                rx_index += 1
            calSum = ~calSum & 0xFF
            # print(calSum)
            if calSum != rxpacket[rx_index]:
                return None, CommResult.RX_CORRUPT

            return RetData(error=Error, params=ret_params), CommResult.SUCCESS
        # print(rx_index)
        return None, CommResult.RX_CORRUPT

    def isAvailable(self, scs_id, address, data_length): #->Tuple[bool, int]:
        #if self.last_result is False or scs_id not in self._recv_data_dict:
        if scs_id not in self._rcv_data_dict:
            return False #, 0

        if (address < self.start_address) or (self.start_address + self.data_length - data_length < address):
            return False #, 0
        if self._rcv_data_dict[scs_id] is None:
            return False #, 0
        if len(self._rcv_data_dict[scs_id].params)<(data_length):
            return False #, 0
        if (_error:= self._rcv_data_dict[scs_id].error) != 0:
            # TODO: handle error.
            raise ValueError(f"actuator id: {scs_id} error: {_error}")
            return False
        return True  #, self._recv_data_dict[scs_id].error

    # def getDataAsInt(self, scs_id, address, data_length)->int:
    #     assert  data_length <= 4 and data_length < self.data_length
    #     assert address >= self.start_address
    #     offset = address - self.start_address
    #     return int.from_bytes(
    #         self._recv_data_dict[scs_id].params[offset: offset+data_length],
    #         byteorder='little')

        # if data_length == 1:
        #     return self._recv_data_dict[scs_id][address-self.start_address+1]
        #
        # elif data_length == 2:
        #     return self.packet_handler.scs_makeword(self._recv_data_dict[scs_id][address-self.start_address+1],
        #                         self._recv_data_dict[scs_id][address-self.start_address+2])
        # elif data_length == 4:
        #     return self.packet_handler.scs_makedword(self.packet_handler.scs_makeword(self._recv_data_dict[scs_id][address-self.start_address+1],
        #                                       self._recv_data_dict[scs_id][address-self.start_address+2]),
        #                          self.packet_handler.scs_makeword(self._recv_data_dict[scs_id][address-self.start_address+3],
        #                                       self._recv_data_dict[scs_id][address-self.start_address+4]))
        # else:
        #     raise ValueError(f'GroupSyncReader can not support getDataAsInt with data_length > 4, {data_length=:}')


    def getDataAsBytes(self, *, scs_id:int, address:int, size:int) -> bytearray:
        assert size <= self.data_length
        assert address >= self.start_address
        offset = address - self.start_address
        # NOTE: copy to return new bytearray object instead of slicing.
        return self._rcv_data_dict[scs_id].params[offset: offset + size].copy()
