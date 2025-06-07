#!/usr/bin/env python

from typing import NoReturn, Dict
from copy import deepcopy
from .scservo_def import *
from .protocol_packet_handler import ProtocolPacketHandler

class GroupSyncWriter:
    def __init__(self,*, packet_handler:ProtocolPacketHandler, start_address, data_length):
        param:bytearray
        write_data_dict:Dict[int, bytearray]
        
        self.packet_handler = packet_handler
        self.start_address = start_address
        self.data_length = data_length
        self.is_param_changed = False
        
        # NOTE: write_data_dict,param are set/cleared at every sync_write.
        self.param = bytearray()
        self.write_data_dict = {}
        

    def makeParam(self)->bool:
        if len(self.write_data_dict) == 0:
            assert False
            return False

        self.param.clear()

        for _id, _data in self.write_data_dict.items():
            if len(_data) == 0:
                assert False
                return False    

            self.param.append(_id)
            self.param.extend(_data) # write_data_dict[scs_id] is copied from external, so it is safe to not copy here. 
            
        return True

    # NOTE: write_data_dict,param are set/cleared at every sync_write. 
    def addParam(self, scs_id, data: bytes|bytearray )->bool:
        if scs_id in self.write_data_dict:  # scs_id already exist
            return False

        if len(data) > self.data_length:  # input data is longer than set
            return False

        # NOTE: make a copy to avoid the buffer of data will be modified after addParam, 
        # cause we don't send the packet immediately after addParam, instead, complete 
        # collecting  all the data from all ID.
        self.write_data_dict[scs_id] = deepcopy(data) # data.copy()

        self.is_param_changed = True
        return True

    def removeParam(self, scs_id)->NoReturn:
        if scs_id not in self.write_data_dict:  # NOT exist
            return

        del self.write_data_dict[scs_id]

        self.is_param_changed = True

    # def changeParam(self, scs_id, data:bytes|bytearray)-> bool:
    #     if scs_id not in self.write_data_dict:  # NOT exist
    #         return False
    # 
    #     if len(data) > self.data_length:  # input data is longer than set
    #         return False
    # 
    #     # NOTE: make a copy to avoid the buffer of data will be modified after addParam, 
    #     # cause we don't send the packet immediately after addParam, instead, complete 
    #     # collecting  all the data from all ID.
    #     self.write_data_dict[scs_id] = data.copy()
    # 
    #     self.is_param_changed = True
    #     return True

    def clearParam(self):
        self.write_data_dict.clear()
        self.param.clear()

    def txPacket(self)-> CommResult:
        if len(self.write_data_dict.keys()) == 0:
            return CommResult.NOT_AVAILABLE

        if self.is_param_changed is True or len(self.param)==0:
            if not self.makeParam():
                assert False
                return CommResult.TX_ERROR                

        return self.packet_handler.syncWriteTxOnly(self.start_address, self.data_length, self.param,
                                       len(self.write_data_dict.keys()) * (1 + self.data_length)) # 1 is for ID.
