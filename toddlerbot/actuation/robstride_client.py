
import atexit
import time
from enum import Enum
from typing import (Any, Dict, List, Optional, Sequence,Type,Mapping,
                    Set, Tuple, ClassVar,Iterable,NamedTuple,Callable, OrderedDict)

import numpy as np
import numpy.typing as npt
from can.interfaces.pcan import *
import can

from .robstride_sdk import *
from ._module_logger import logger

# @dataclass(init=False)
class RobStrideClient:
    """Client for communicating with a group of Feite motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients. class variable.
    OPEN_CLIENTS: ClassVar[Set[Any] ] = set()

    # instance variable.
    bus: PcanBus | None

    def __init__(
        self,*,
        motor_ids: Sequence[int],  # ids of a group of actuators.
        interface_name: str,             #= "can0",
        baud_rate: int,             # = 115200,  # default for SMS/STS series.
        lazy_connect: bool,          #= False,
        rcv_timeout_ms: int,              #= 5,    #usb serial latency timer, default 5 ms.
    ):
        """Initializes a new client.

        Args:
        """

        """
           初始化CAN电机控制器。

           参数:
           bus: CAN总线对象。
           motor_id: 电机的CAN ID。
           main_can_id: 主CAN ID。
           """

        # NOTE: _motor_ids is not guaranteed to be consecutive, i.e, could be [44, 1, 230,...]. so we
        # could not use id as array index directly.
        self._motor_ids = motor_ids  # not changed after instantiating a FeiteClient instance.
        self.interface_name = interface_name
        self.baud_rate = baud_rate
        self.lazy_connect = lazy_connect
        self.rcv_timeout_ms = rcv_timeout_ms
        self.bus = None

        RobStrideClient.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    def connect(self):
        """Connects to the motors.

        NOTE: This should be called after all RobstrideClients on the same
            process are created.
        """
        assert not self.is_connected, "Client is already connected."

        self.bus = PcanBus(channel=,
                           device_id=,
                           state=,
                           timing=,
                           bitrate=self.baud_rate,
                           receive_own_messages=,)

        if self.bus.status_is_ok():
            logger.info(f"Succeeded to open can interface: {self.interface_name}"
                        f",with baud_rate:{self.baud_rate}")
        else:
            raise OSError(
                (
                    "Failed to open can interface at {}, bus status:{} (Check that the device is powered "
                    "on and connected to your computer)."
                ).format(self.interface_name,self.bus.status_string())
            )

        # Start with all motors enabled.  NO, I want to set settings before enabled
        # self.set_torque_enabled(self._motor_ids, True)

    def disconnect(self):
        """Disconnects from the RobStride motors."""
        if not self.is_connected:
            return
        # if self.port_handler.is_using:
        #     logger.error("Port handler in use; cannot disconnect.")
        #     return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(motor_ids=self._motor_ids, enabled=False)
        self.bus.shutdown()
        self.bus = None

        if self in RobStrideClient.OPEN_CLIENTS:
            RobStrideClient.OPEN_CLIENTS.remove(self)

    def check_connected(self):
        """Ensures the motor is connected."""
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError("Must call connect() first.")

    def _send_recv_can_message(self, *,
                                 motor_can_id: int,
                                 comm_type: int,
                                 data2: int,
                                 data1: bytes | bytearray,
                                 timeout_sec=0.01) \
        ->Tuple[bytearray, int]:
        """
        发送CAN消息并接收响应。

        参数:
        cmd_mode: 命令模式。
        data2: 数据区2。
        data1: 要发送的数据字节。
        timeout: 发送消息的超时时间(默认为10ms)。

        返回:
        一个元组, 包含接收到的消息数据和接收到的消息仲裁ID(如果有)。
        """
        assert 0 <= motor_can_id <= 0xff  # 1-byte
        assert 0 < comm_type <= 0b11111   # 5-bit
        assert 0 < data2 <= 0xffff        # 2-bytes
        assert len(data1) == 8            # always 8-bytes

        # Calculate the arbitration ID
        arbitration_id = (comm_type << 24) | (data2 << 8) | motor_can_id
        try:
            # will check arbitration ID in can.Message.
            snd_msg = can.Message(
                arbitration_id=arbitration_id,
                dlc=8,
                data=data1,
                is_extended_id=True
            )
        except Exception as exc:
            logger.error(f'build can msg error: {exc=:} {type(exc)=:}')
            raise exc

        logger.debug(
            f"Sent message with ID {hex(arbitration_id)}, data: {data1}")

        # Send the CAN message
        try:
            self.bus.send(snd_msg)
        except PcanError as exc:
            logger.error(f"Failed to send the can message, {exc=:} {type(exc)=:}")
            raise exc

        # TODO: use The Notifier object is used as a message distributor for a bus.
        #  The Notifier uses an event loop or creates a thread to read messages
        #  from the bus and distributes them to listeners.
        # or use the asyncio.
        # 10ms timeout for receiving
        try:
            rcv_msg = self.bus.recv(timeout=timeout_sec)
        except can.CanError as exc:
            logger.error(f"Failed to recv the can message, {exc=:} {type(exc)=:}")
            raise exc

        if rcv_msg is not None:
            assert rcv_msg.is_extended_id
            return rcv_msg.data, rcv_msg.arbitration_id
        else:
            raise IOError(f'recv can msg timeout.')

    def set_torque_limit(
        self, *,
        motor_ids: Sequence[int],
        max_torque: npt.NDArray[np.float32],
        retries: int = -1,
        retry_interval: float = 0.25,
    ):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids:
            max_torque: array.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        assert np.all(max_torque > 0)
        # uint_max_torque = ParamConverter.float_normalized_to_uint(x=max_torque,
        #                                         x_min=0.,
        #                                         # TODO: this is for RS02 only.
        #                                         x_max=ParamThreshold.T_MAX,
        #                                         n_bytes=4)

        limit_dict:OrderedDict[int, npt.NDArray[np.float32]] = OrderedDict( zip(motor_ids,
                                                                               max_torque) )

        logger.info(f'set torque limit: {limit_dict}')

        remaining_id = [*limit_dict.keys()] #list(motor_ids)
        while len(remaining_id) > 0 :
            # guarantee order.

            limit_bytes: List[bytes] = [ ParamConverter.float_to_param_bytes(limit_dict[_id])
                                        for _id in remaining_id ] # [ *remaining_ids.values() ]

            remaining_id = self._write_and_recv_answer_impl(param=limit_bytes,
                                  address=SMS_STS_SRAM_Table_RW.TORQUE_LIMIT_L,
                                  write_ids=remaining_id)

            if len(remaining_id) > 0:
                logger.error(f"Could not set torque limit for IDs: {remaining_id}. maybe retry.")
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1


    def set_torque_enabled(
        self, *,
        motor_ids: Sequence[int],
        enabled: bool,
        retries: int = -1,
        retry_interval: float = 0.25,
    ):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        remaining_ids = motor_ids #list(motor_ids)
        while remaining_ids:
            remaining_ids = self._write_and_recv_answer_impl(param=int(enabled).to_bytes(length=1),
                                                             address=SMS_STS_SRAM_Table_RW.TORQUE_ENABLE,
                                                             write_ids=remaining_ids )

            if remaining_ids:
                logger.error(f"Could not set torque {'enabled' if enabled else 'disabled'} for IDs: {str(remaining_ids)}")
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def _sync_read_helper(self,read_spec:_ReadSpec )\
            ->Tuple[float, Sequence[npt.NDArray[Any]] ]:
        param_list: List[bytearray]
        value_arr_list: List[npt.NDArray[Any]] = []

        assert len(read_spec.size) ==  len(read_spec.parser) == len(read_spec.result_dtype)

        for _dtype in read_spec.result_dtype:
            if _dtype in [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32]:
                fill_value = 0
            elif _dtype in [np.float16, np.float32, np.float64]:
                fill_value = np.nan
            else:
                raise TypeError(f'read_spec result dtype error: {_dtype}')
            value_arr_list.append(np.full(shape=len(self._motor_ids), fill_value=fill_value, dtype=_dtype))

        comm_time, param_list = self._sync_read_impl(address=read_spec.start_addr,
                                                     size=sum(read_spec.size))
        assert len(param_list) == len(self._motor_ids)
        # TODO: corresponding data of error id in param_list keeps as `None`. check the valid value ?
        for _x, _bytes in enumerate(param_list):
            if _bytes is None:
                # TODO: propagate nan to up layer caller. _value is `fill_value`, 0 or np.nan.
                raise ValueError(f'the read data of ID:{self._motor_ids[_x]} is None.')
            else:
                # if sum(read_spec.size) == 6:
                #     logger.info(f'--- sync read bytes:')
                #     for _b in _bytes:
                #         print(f'0x{_b:02x}')
                #     logger.info(f'end of bytes --- ')

                assert len(_bytes) == sum(read_spec.size)
                for _s, _psr, _arr in zip(read_spec.size, read_spec.parser, value_arr_list):
                    # bytearray(_bytes.pop(0) for _ in range(_s))
                    _arr[_x] = _psr(_bytes[0:_s])
                    del _bytes[0:_s]

        # NOTE: if the ID does not return pkt, the corresponding value is `0`.
        return comm_time, value_arr_list

    def _read_table_single_value_helper(self,*, name: TableValueName, into_cache: bool, retries: int = 0) \
            -> Tuple[float,npt.NDArray[Any]]:
        comm_time, value_arr_list = self._sync_read_helper(_TableValueReadSpec[name])
        assert len(value_arr_list)==1 and len(value_arr_list)==len(_TableValueReadSpec[name].result_dtype)
        if into_cache:
            assert self._cached_read_data_dict[name].dtype == _TableValueReadSpec[name].result_dtype[0]
            self._cached_read_data_dict[name] = value_arr_list[0].copy()

        # NOTE: if the ID does not return pkt, the corresponding value is `np.nan`.
        return comm_time, value_arr_list[0]

    def _read_table_multiple_values_helper(self,*, name: TableValueName, retries: int = 0)\
            -> Tuple[float,Sequence[npt.NDArray[Any]] ]:
        comm_time, value_arr_list = self._sync_read_helper(_TableValueReadSpec[name])
        assert len(value_arr_list) > 1 and len(value_arr_list) == len(_TableValueReadSpec[name].result_dtype)
        # if into_cache:
        #     assert self._cached_read_data_dict[name].dtype == _TableValueReadSpec[name].result_dtype
        #     self._cached_read_data_dict[name] = value.copy()

        # NOTE: if the ID does not return pkt, the corresponding value is `np.nan`.
        return comm_time, value_arr_list


    def read_model_number(self, retries: int = 0) -> npt.NDArray[np.uint16]:
        _, ret = self._read_table_single_value_helper(name=TableValueName.model, into_cache=False)
        return ret

    def read_pos(self, retries: int = 0) -> Tuple[float,npt.NDArray[np.float32]]:
        return self._read_table_single_value_helper(name=TableValueName.pos, into_cache=True)


    def read_vel(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        return self._read_table_single_value_helper(name=TableValueName.vel, into_cache=True)

     def read_vin(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float16]]:
        return self._read_table_single_value_helper(name=TableValueName.vin, into_cache=True)

     def read_pos_vel_load(
        self, retries: int = 0
    ) -> PosVelLoadRead:
        comm_time, value_arr_list = self._read_table_multiple_values_helper(name=TableValueName.pos_vel_load)

        # TODO: the necessary of saving into cached data dict?
        self._cached_read_data_dict[TableValueName.pos]=value_arr_list[0]
        self._cached_read_data_dict[TableValueName.vel]=value_arr_list[1]
        self._cached_read_data_dict[TableValueName.load]=value_arr_list[2]  # load in percentage of stall torque.

        return PosVelLoadRead(
            comm_time,
            pos=self._cached_read_data_dict[TableValueName.pos].copy(),
            vel=self._cached_read_data_dict[TableValueName.vel].copy(),
            load=self._cached_read_data_dict[TableValueName.load].copy(),
        )

    def read_protect_mode(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.uint8]]:
            return self._read_table_single_value_helper(name=TableValueName.protect_mode,
                                                        into_cache=False)

    def _sync_write_impl(
        self,*,
        param: Sequence[bytes|bytearray] | bytes| bytearray,
        address: int,
        # size: int,
        write_ids: Optional[Sequence[int]] = None,
    ):
        """Writes values to a group of motors. no answer pkt.

        Args:
            write_ids: The motor IDs to write to.
            param: The values to write. single bytes/bytearray value means same value for all ID, a list(bytes|bytearray) contains
                   individual value for every ID.
            address: The control table address to write to.
            #size: The size(in bytes) of the control table value being written to. can cover multiple registers.
        """
        size: int
        errored_ids: List[int] = []

        if isinstance(param,(list, tuple)):
            size = len(param[0])  # should be same for all elements in params.
        elif isinstance(param, (bytes, bytearray)):
            size = len(param)
        else:
            raise TypeError(f'param type error: {type(param)} ')

        self.check_connected()
        key = (address, size)

        if key not in self._sync_writers:
            self._sync_writers[key] = GroupSyncWriter(
                packet_handler = self.packet_handler,
                start_address=address,
                data_length=size)

        sync_writer = self._sync_writers[key]

        if write_ids is None:
            write_ids = self._motor_ids

        if isinstance(param, (list,tuple)):
            assert len(write_ids) == len(param)

        # Clear before addParam.
        sync_writer.clearParam()

        if isinstance(param,(list, tuple)):
            for _id, _bytes in zip(write_ids, param):
                assert len(_bytes)==size # should be same for all elements in params.
                success = sync_writer.addParam(_id, _bytes)
                if not success:
                    errored_ids.append(_id)
        else:
            for _id in write_ids:
                # add same value for all ID.
                success = sync_writer.addParam(_id, param)
                if not success:
                    errored_ids.append(_id)

        if errored_ids:
            logger.error( f"Sync write failed for: {str(errored_ids)}"    )

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result, context="sync_write")

        # write_data_dict,param are set/cleared at every sync_write.
        # sync_writer.clearParam()


    def _write_and_recv_answer_impl(
        self,*,
        param: Sequence[bytes|bytearray] | bytes | bytearray, # each element is for corresponding id.
        address: int,    # same for all ids.
        write_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[int]:
        """Writes a value to the motors.
           vs. sync_write:  we need the individual feedback of corresponding ID here, but sync_write has no feedback pkt.

        Args:
            write_ids: The motor IDs to write to.
            param: The value to write to the control table.
            address: The control table address to write to. same for all id.

        Returns:
            A list of IDs that were unsuccessful.
        """
        size: int
        errored_ids: List[int] = []

        if isinstance(param, (list, tuple)):
            size = len(param[0])  # should be same for all elements in params.
        elif isinstance(param, (bytes, bytearray)):
            size = len(param)
        else:
            raise TypeError(f'param type error: {type(param)} ')

        self.check_connected()

        if write_ids is None:
            write_ids = self._motor_ids

        write_data: List[bytes | bytearray]
        if isinstance(param, (list,tuple)):
            assert len(write_ids) == len(param)
            write_data = param
        else:
            write_data = [param] * len(write_ids)

        for _id, _bytes in zip(write_ids, write_data):
            assert len(_bytes) == size  # should be same for all elements in params.
            comm_result,error = self.packet_handler.writeTxRx(
                scs_id = _id,
                address = address,
                length = len(_bytes) ,
                data = _bytes)
            success = self.handle_packet_result(
                comm_result,
                error,
                _id,
                context="_write_and_recv_answer_impl",
            )
            if not success:
                errored_ids.append(_id)


        return errored_ids


    def _set_1_byte_param_helper(self, *, address:int, value:int|Sequence[int]):
        param: bytes | List[bytes]

        if isinstance(value, (list, tuple)):
            for _v in value:
                assert 0 <= _v <= 0xff

            param = [_v.to_bytes(length=1) for _v in value]
        elif isinstance(value, int):
            assert 0 <= value <= 0xff
            param = value.to_bytes(length=1)
        else:
            raise TypeError(f'value type error: {type(value)} ')

        self._sync_write_impl(param=param,
                              address=address)


    def _set_multiple_bytes_param_helper(self,*, address:int, value: Tuple[int,...] | Sequence[Tuple[int,...]]):
        """
        value: each `int` represent one `byte` in memory tbl of actuator.
        """
        param: bytes| List[bytes]

        # if isinstance(value, list) and isinstance(value[0], tuple):
        if isinstance(value[0], tuple):
            for _tpl in value:
                assert isinstance(_tpl, Iterable)
                for _v in _tpl:
                    assert 0 <= _v <= 0xff

            param = [bytes(_tpl) for _tpl in value]

        elif isinstance(value, tuple):
            for _v in value:
                assert 0 <= _v <= 0xff

            param = bytes(value)

        else:
            raise TypeError(f'value type must be tuple(int) or list( tuple(int) ). got:{type(value)}')

        self._sync_write_impl(param=param,
                              address=address)

    def set_goal_pos(self, *, motor_ids: Sequence[int], pos: npt.NDArray[np.float32]):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            pos: The joint angles in radians to write. in rad of single turn.signed value, to represent rotor direction.
        """
        assert len(motor_ids) == len(pos)
        # TODO: only allow -2Pi ~ 2Pi.
        if not np.all(np.abs(pos) < 2 * np.pi):
            raise ValueError(f'not allowed goal pos: {pos}, which should be in [-2pi, 2pi] ')

        # ->steps, ->int, signed->unsigned, to_bytes.
        # Convert to Feite position steps:
        steps = (pos / POS_RESOLUTION).astype(dtype=np.int16)
        size = steps.dtype.itemsize
        assert size==2

        self._sync_write_impl(param=[signed_to_proto_param_bytes_v2(value=int(_v), size=size)
                                          for _v in steps],  # addr 42, 43,
                              address=SMS_STS_SRAM_Table_RW.GOAL_POSITION_L,
                              write_ids=motor_ids)

    def set_goal_accel(self, *, motor_ids: Sequence[int], accel: npt.NDArray[np.float32]):
            """Writes the given accel.

            Args:
                motor_ids: The motor IDs to write to.
                accel:
            """
            assert len(motor_ids) == len(accel)

            # if  np.any(accel < 0) or np.any(accel > 3 * np.pi):
            if False:
                raise ValueError(f'not allowed goal accel: {accel}, which should be in [0, 3pi] ')

            # ->steps, ->int, signed->unsigned, to_bytes.
            # Convert to Feite position steps:
            steps = (accel / ACCEL_RESOLUTION).astype(dtype=np.uint8)
            size = steps.dtype.itemsize
            assert size == 1

            self._sync_write_impl(param=[int(_v).to_bytes(length=size, byteorder='little', signed=False)
                                         for _v in steps],  # addr 41,
                                  address=SMS_STS_SRAM_Table_RW.GOAL_ACCEL,
                                  write_ids=motor_ids)

    def set_goal_vel(self, *, motor_ids: Sequence[int], vel: npt.NDArray[np.float32]):
        """Writes the given vel.

        Args:
            motor_ids: The motor IDs to write to.
            vel:
        """
        assert len(motor_ids) == len(vel)

        # if np.any (abs(vel) > 3 * np.pi/2 ):
        if False:
            raise ValueError(f'not allowed goal vel: {vel}, which should be in [-3pi/2, 3pi/2] ')

        # ->steps, ->int, signed->unsigned, to_bytes.
        # Convert to Feite position steps:
        steps = (vel / VEL_RESOLUTION).astype(dtype=np.int16)
        size = steps.dtype.itemsize
        assert size == 2

        self._sync_write_impl(param=[signed_to_proto_param_bytes_v2(value=int(_v), size=size)
                                     for _v in steps],  # addr 46,
                              address=SMS_STS_SRAM_Table_RW.GOAL_VEL_L,
                              write_ids=motor_ids)

        # self.sync_write_2_bytes(_motor_ids=_motor_ids,
        #                         params=[_signed_to_proto_param_bytes_v2(value=_v, size=2) for _v in steps],  # addr 42, 43,
        #                         address=SMS_STS_SRAM_Table_RW.GOAL_POSITION_L)  # addr 42, 43


    def set_return_delay_time(self, value: int|Sequence[int]):
        # This sync writing section has to go after the voltage reading to make sure the motors are powered up
        # Set the return delay time to value us. EEPROM Table value unit is 2 us, so we //2 here.
        assert value >= 2
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.RETURN_DELAY_TIME, value=value // 2)

    def set_control_mode(self, value: int|Sequence[int]):
        # TODO: only allow 0,1,2,3
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.CONTROL_MODE,value=value)

    def set_kp(self, value: int | Sequence[int]):
        #TODO: not allow 0 value for kp....
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.KP, value=value)

    # def set_kp_kd(self, *, kp: int | Seq[int], kd: int|List[int]):
    #     assert 0 < kp < 0xff
    #     assert 0 < kd < 0xff
    #
    #     # kp:21, kd:22 consecutive address.
    #     self._sync_write_impl(param=bytes([kp, kd]),
    #                           address=SMS_STS_EEPROM_Table_RW.KP)

    def set_kp_kd_ki(self, *, kp: Sequence[int], kd: Sequence[int], ki: Sequence[int]):
        # kp:21, kd:22 , ki:23 consecutive address.
        self._set_multiple_bytes_param_helper(address=SMS_STS_EEPROM_Table_RW.KP,
                                              value=list(zip(kp, kd, ki)) )

    def __enter__(self):
        """Enables use as a context manager."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.disconnect()


def _client_cleanup_handler():
    """Handles cleanup of open Feite clients by forcibly closing active connections.

    Iterates over all open Feite clients and checks if their port handlers are in use.
    If a port handler is active, logs a warning message and forces the client to close
    by setting the port handler's `is_using` attribute to False and disconnecting the client.
    """
    open_clients: List[RobStrideClient] = list(RobStrideClient.OPEN_CLIENTS)  # type: ignore
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logger.warning("Forcing client to close.")
        open_client.port_handler.is_using = False
        open_client.disconnect()

# Register global cleanup function.
atexit.register(_client_cleanup_handler)
