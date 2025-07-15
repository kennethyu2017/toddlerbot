from typing import (List, Type, NamedTuple)
from numpy import typing as npt
import numpy as np
from can import Message

from .._module_logger import logger
from .robstride_def import (CommunicationType, BaudRateCmd, ParamThreshold,
                            param_table_index_to_name, RS_param_table_spec,
                            MotorStateFrame, ExtID , SingleParamValue, ParamSpec)

from .utils import ParamConverter

# for send msg.
class RSProtocolBuilder:

    @staticmethod
    def _can_msg_helper(*, motor_can_id: npt.NDArray[np.uint32],
                        comm_type: int,
                        data2: npt.NDArray[np.uint32] | int,
                        data1: List[bytes|bytearray|None] )->List[Message]:

        assert np.all(0 <= motor_can_id) and np.all(motor_can_id <= 0xff)  # 1-byte
        assert np.all(0 <= comm_type) and np.all(comm_type <= 0b11111)  # 5-bit
        assert np.all(0 <= data2) and np.all(data2 <= 0xffff)
        assert len(motor_can_id) == len(data1)

        arbitration_id = (comm_type << 24) | (data2 << 8) | motor_can_id
        msg_list: List[Message] = []
        for _id, _d in zip(arbitration_id, data1):
            #TODO: try variable DLC. 8-bytes data zone.
            # assert len(_d) == 8

            try:
                # will check arbitration ID in can.Message.
                msg_list.append(Message(
                    arbitration_id=_id,
                    # dlc=8,
                    dlc=None, # make API calc dlc.
                    data=_d,
                    is_extended_id=True)
                )
            except Exception as exc:
                logger.error(f'build can msg error: {exc=:} {type(exc)=:}')
                raise exc

        return msg_list


    # @staticmethod
    # def build_arbitration_id(*, dest: npt.NDArray[np.uint32] | int,
    #                          comm_type: npt.NDArray[np.uint32] | int,
    #                          data2:npt.NDArray[np.uint32] | int )->npt.NDArray[np.uint32]:
    #
    #     assert np.all(0 <= dest) and np.all(dest <= 0xff)  # 1-byte
    #     assert np.all(0 <= comm_type) and np.all(comm_type <= 0b11111)    # 5-bit
    #     assert np.all(0 <= data2) and np.all(data2 <= 0xffff)
    #
    #     return (comm_type << 24) | (data2 << 8) | dest


    @staticmethod
    def get_device_id(motor_can_id: npt.NDArray[np.uint32], host_can_id: int) -> List[Message]:
        assert 0 <= host_can_id <= 0xfe

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.GET_DEVICE_ID,
                                                 data2=host_can_id,
                                                 # data1=[bytes(8)] * len(motor_can_id)
                                                 data1=[None] * len(motor_can_id) )

    @staticmethod
    def motion_control():
        pass

    @staticmethod
    def motor_enable(motor_can_id: npt.NDArray[np.uint32], host_can_id:int)->List[Message]:
        assert 0 <= host_can_id <= 0xfe

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.MOTOR_ENABLE,
                                                 data2=host_can_id,
                                                 # data1=[bytes(8)] * len(motor_can_id)
                                                 data1=[None] * len(motor_can_id)
                                                 )

    @staticmethod
    def motor_disable(motor_can_id: npt.NDArray[np.uint32], host_can_id:int, clear_error:bool = False)\
            ->List[Message]:
        assert 0 <= host_can_id <= 0xfe

        # data1 = bytearray(8)  # inited as all null bytes.
        # if clear_error:
        #     data1[0] = 1

        cmd: int = 1 if clear_error else 0
        data1 = [bytes([cmd])] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.MOTOR_DISABLE,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )


    @staticmethod
    def set_mech_pos_zero(motor_can_id: npt.NDArray[np.uint32], host_can_id:int)\
            ->List[Message]:
        assert 0 <= host_can_id <= 0xfe
        # data1 = bytearray(8)  # inited as all null bytes.
        data1 = [bytes([1])] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.SET_MECH_POS_ZERO,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )

    @staticmethod
    def set_motor_can_id(motor_can_id: npt.NDArray[np.uint32],
                         host_can_id:int,
                         new_can_id: npt.NDArray[np.uint32])->List[Message]:
        assert 0 <= host_can_id <= 0xfe
        assert np.all(0<=new_can_id) and np.all(new_can_id<=0xfe)

        data2 = (new_can_id << 8) | host_can_id
        data1 = [None] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.SET_MOTOR_CAN_ID,
                                                 data2=data2,
                                                 # data1=[bytes(8)] * len(motor_can_id)
                                                 data1=data1
                                                 )

    @staticmethod
    def read_single_param(motor_can_id: npt.NDArray[np.uint32],
                          host_can_id:int,
                          index:int)->List[Message]:
        assert 0 <= host_can_id <= 0xfe
        assert 0x7005 <= index <= 0x7029

        # data1 = bytearray(8)
        # data1[0], data1[1]=index.to_bytes(length=2, byteorder='little',signed=False)
        # ignore byte2~7
        idx_bytes=index.to_bytes(length=2, byteorder='little',signed=False)
        data1 = [idx_bytes] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.SINGLE_PARAM_READ,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )

    @staticmethod
    def write_single_param(motor_can_id: npt.NDArray[np.uint32],
                           host_can_id: int,
                           index: int,
                           param_value: List[float|int],
                           param_spec: ParamSpec
                           )->List[Message]:
        assert len(param_value) == len(motor_can_id)
        assert 0 <= host_can_id <= 0xfe
        assert 0x7005 <= index <= 0x7029
        assert param_spec.n_bytes in {1,2,4}
        assert param_spec.dtype in {int,float}

        idx_bytes = index.to_bytes(length=2, byteorder='little', signed=False)

        data1_lst: List[bytes] = []

        for _v in param_value:
            # data1 = bytearray(8)
            data1 = bytearray(4)
            data1[0], data1[1] = idx_bytes

            p_bytes = ParamConverter.value_to_param_bytes(value=_v,
                                                n_bytes=param_spec.n_bytes,
                                                dtype=param_spec.dtype,
                                                signed=param_spec.signed)
            assert len(p_bytes) == param_spec.n_bytes and len(p_bytes)<=4

            # data1[4:4+len(p_bytes)] = p_bytes
            data1.extend(p_bytes)

            data1_lst.append(data1)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                         comm_type=CommunicationType.SINGLE_PARAM_WRITE,
                                         data2=host_can_id,
                                         data1=data1_lst)


    @staticmethod
    def save_motor_param(motor_can_id: npt.NDArray[np.uint32], host_can_id: int)\
            ->List[Message]:
        assert 0 <= host_can_id <= 0xfe

        data1 = [bytes([1, 2, 3, 4, 5, 6, 7, 8])] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.SAVE_PARAM,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )
    @staticmethod
    def set_baud_rate(motor_can_id: npt.NDArray[np.uint32], host_can_id: int,
                      baud_rate_cmd: int) ->List[Message]:
        assert 0 <= host_can_id <= 0xfe
        assert baud_rate_cmd in {BaudRateCmd.BPS_1M, BaudRateCmd.BPS_500K,
                                 BaudRateCmd.BPS_250K, BaudRateCmd.BPS_125K}

        data1 = [bytes([1, 2, 3, 4, 5, 6, baud_rate_cmd])] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.SET_BAUD_RATE,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )

    @staticmethod
    def set_motor_periodic_report(motor_can_id: npt.NDArray[np.uint32], host_can_id: int,
                                  enable:bool) ->List[Message]:
        assert 0 <= host_can_id <= 0xfe

        cmd: int = 1 if enable else 0
        data1 = [bytes([1, 2, 3, 4, 5, 6, cmd])] * len(motor_can_id)

        return RSProtocolBuilder._can_msg_helper(motor_can_id=motor_can_id,
                                                 comm_type=CommunicationType.MOTOR_PERIODIC_REPORT,
                                                 data2=host_can_id,
                                                 data1=data1
                                                 )

# class MotorState(NamedTuple):
#     can_id: int
#     pos: float
#     vel: float
#     torque: float
#     temp_celsius: float
#     #todo: motor error...


# parse rcv msg. one by one parse for recv msg.
class RSProtocolParser:

    @staticmethod
    def decode_ext_id(arbitration_id: int)->ExtID:
        # arbitration_id = (comm_type << 24) | (data2 << 8) | motor_can_id
        dest_can_id = arbitration_id & 0xff
        data2 = (arbitration_id >> 8) & 0xffff
        comm_type = (arbitration_id >> 24) & 0b11111
        return ExtID(dest_can_id=dest_can_id, data2=data2, comm_type=comm_type)

    @staticmethod
    def motor_device_id(data2: int, data: bytes | bytearray) -> int:
        motor_can_id = data2 & 0xff
        assert len(data) == 8
        mcu_id = int.from_bytes(data, byteorder='little', signed=False)
        return mcu_id

    @staticmethod
    def motor_state_feedback(data2:int, data:bytes|bytearray, ts:float )->MotorStateFrame:
        # ext_id = RSProtocolParser.decode_ext_id(msg.arbitration_id)
        motor_can_id = data2 & 0xff
        motor_error = (data2 >> 8) & 0xff

        # TODO: handle all the faults.
        # check bit 16~21
        if (motor_error & 0b111111) != 0:
            raise IOError(f'motor state feedback error: {motor_error=:}')

        # state data is big endian.
        pos:float = ParamConverter.uint_normalized_to_float( #    x=(data[0] << 8) + data[1],
                                                      x=int.from_bytes(data[0:2], byteorder='big', signed=False),
                                                      x_min=ParamThreshold.P_MIN,
                                                      x_max=ParamThreshold.P_MAX,
                                                      n_bytes=2)
        vel:float = ParamConverter.uint_normalized_to_float(  # x= (data[2] << 8) + data[3]
            x=int.from_bytes(data[2:4], byteorder='big', signed=False),
            x_min=ParamThreshold.V_MIN,
            x_max=ParamThreshold.V_MAX,
            n_bytes=2)

        torque:float = ParamConverter.uint_normalized_to_float(  # x= (data[4] << 8) + data[5]
            x=int.from_bytes(data[4:6], byteorder='big', signed=False),
            x_min=ParamThreshold.T_MIN,
            x_max=ParamThreshold.T_MAX,
            n_bytes=2)

        temp_celsius:float = int.from_bytes(data[6:8], byteorder='big', signed=False) / 10.

        return MotorStateFrame(ts=ts,
                               can_id=motor_can_id,
                               pos=pos,
                               vel=vel,
                               torque=torque,
                               temp=temp_celsius,
                               # motor_error...
                               )

    @staticmethod
    def single_param(data2:int, data:bytes|bytearray)->SingleParamValue:
        motor_can_id = data2 & 0xff
        read_success: bool = ((data2 >> 8) & 0xff) == 0

        if read_success:
            # single param read result is in `little-endian`.
            index = int.from_bytes(data[0:2], byteorder='little',signed=False)
            if index not in param_table_index_to_name:
                raise ValueError(f'param index value error: {index=:}')

            p_name: str = param_table_index_to_name[index]
            p_spec = RS_param_table_spec[p_name]
            assert p_spec.index == index
            value:float|int = ParamConverter.param_bytes_to_value(param=data[4:4+p_spec.n_bytes],
                                                n_bytes=p_spec.n_bytes,
                                                dtype=p_spec.dtype,
                                                signed=p_spec.signed)
            return SingleParamValue(can_id=motor_can_id, index=index, value=value)

        else:
            raise IOError(f'read motor single param failed: {motor_can_id=:}')


    @staticmethod
    def error_feedback():
        # comm type : 0x15
        # TODO> implement.
        pass




