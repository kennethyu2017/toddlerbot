import atexit
import time
from typing import (Any, Dict, List, Optional, Sequence,
                    Set, Tuple, ClassVar,Iterable,NamedTuple,
                    OrderedDict)
from dataclasses import dataclass, field
# deque does not use lock, but its append/popleft is atomic operation.
from collections import OrderedDict, deque
import asyncio
from aiologger import Logger
from aioconsole import aprint
# from queue import Queue  # thread-safe fifo-queue.
import numpy as np
import numpy.typing as npt
import struct
import can
from can.interfaces.socketcan import SocketcanBus

from .robstride_sdk import *
# from ._module_logger import logger
alogger = Logger.with_default_handlers()

# @dataclass
# class MotorData:
#     can_id: int = 0
#     # thread-safe sync-fifo-queue.
#     state_queue: deque[MotorStateFrame] = field(default_factory= lambda: deque(maxlen=30)) # cached with ts.
#     param_table: Dict[int, float|int] = field(default_factory=dict) # not cached. only fresh value.

# @dataclass(init=False)
# class MotorData:
#     # index by motor can id.
#     # TODO: protect by lock?
#     state_queue: OrderedDict[int, asyncio.Queue[MotorStateFrame] ]
#     param_queue: OrderedDict[int, asyncio.Queue[SingleParamValue] ]
#
#     def __init__(self):
#         self.state_queue = OrderedDict( (_id, asyncio.Queue(maxsize=30)) for _id in MOTOR_CAN_ID_SET )
#         self.param_queue = OrderedDict( (_id, asyncio.Queue(maxsize=30)) for _id in MOTOR_CAN_ID_SET )
#


# one client for one can bus.
class RobStrideClient:
    """Client for communicating with a group of Feite motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients. class variable.
    OPEN_CLIENTS: ClassVar[Set[Any] ] = set()

    # instance variable.
    bus: SocketcanBus | None
    # motor_data: OrderedDict[int, MotorData]
    # index by motor can id.
    # TODO: protect by lock?
    motor_state_q: OrderedDict[int, deque[MotorStateFrame] ]

    # not use queue.
    motor_param_table: OrderedDict[int, Dict[int, SingleParamValue|None] ]

    _send_buffer_q: asyncio.Queue[can.Message]
    _rcv_buffer_q: asyncio.Queue[can.Message]

    def __init__(
        self,*,
        motor_can_id: Sequence[int],  # ids of a group of actuators.
        host_can_id: int,     # 0xfe
        channel_name: str,             #= "can0",
        baud_rate: int,             # = 115200,  # default for SMS/STS series.
        lazy_connect: bool,          #= False,
        rcv_timeout_ms: int,              #= 5,    #usb serial latency timer, default 5 ms.
    ):
        """Initializes a new client.
        Args:
            motor_can_id:
            host_can_id:
            channel_name:
            baud_rate:
            lazy_connect:
            rcv_timeout_ms:
           """

        # NOTE: _motor_ids is not guaranteed to be consecutive, i.e, could be [44, 1, 230,...]. so we
        # could not use id as array index directly.
        self._motor_can_id = motor_can_id  # not changed after instantiating a FeiteClient instance.
        self.channel_name = channel_name
        assert 0x7f < host_can_id <= 0xfe
        self.host_can_id = host_can_id
        self.baud_rate = baud_rate
        self.lazy_connect = lazy_connect
        self.rcv_timeout_ms = rcv_timeout_ms
        self.bus = None

        # index through motor can id.
        # self.motor_data: OrderedDict[int, MotorData] = OrderedDict()
        # TODO: adjust queue size according to control freq.
        self.motor_state_q = OrderedDict( (_id, deque(maxlen=10)) for _id in motor_can_id)
        self.motor_param_table = OrderedDict((_id, dict()) for _id in motor_can_id)

        # TODO: protected by lock.
        self._rcv_buffer_q: asyncio.Queue[can.Message] = asyncio.Queue(maxsize=10 * len(motor_can_id))
        self._send_buffer_q: asyncio.Queue[can.Message] = asyncio.Queue(maxsize=10 * len(motor_can_id))

        RobStrideClient.OPEN_CLIENTS.add(self)

    # NOTE: callback invoked in running_loop: do not implement as co-routine directly:
    def _on_read_available(self) -> None:
        if msg := self.bus.recv(timeout=0):
            # if buffer q is full, raise error directly.
            try:
                # append/popleft op is atomic, no need to add lock.
                self._rcv_buffer_q.put_nowait(msg)
            except Exception as exc:
                print(f'put rcv msg to buffer failed: {exc=:},  {type(exc)=:}. maybe increase buffer size.')
                raise

    # NOTE: callback invoked in running_loop: do not implement as co-routine directly:
    def _on_write_available(self) -> None:
        if not self._send_buffer_q.empty():
            try:
                msg = self._send_buffer_q.get_nowait()
                self.bus.send(msg, timeout=0)
                self._send_buffer_q.task_done()

            except Exception as exc:
                raise OSError(f'send msg from buffer queue failed: {exc=:} {type(exc)=:}')

    async def _parse_rcv_msg(self) -> None:
        while True:
            try:
                msg: can.Message = await self._rcv_buffer_q.get()

                await alogger.debug(f'rcv msg: {msg}')

                ext_id = RSProtocolParser.decode_ext_id(msg.arbitration_id)

                # filtered by socket can.
                # if ext_id.dest_can_id != self.host_can_id:
                #     raise ValueError(f'rcv msg dest can id is not host, dest id: {ext_id.dest_can_id}')

                await alogger.debug(f'parse msg result--->')
                if ext_id.comm_type == CommunicationType.SINGLE_PARAM_READ:
                    param_value = RSProtocolParser.single_param(data2=ext_id.data2, data=msg.data)
                    await alogger.debug(f'{param_value}')

                    p_table: Dict[int, SingleParamValue] = self.motor_param_table[param_value.can_id]

                    if param_value.index in p_table and p_table[param_value.index] is not None:
                        raise ValueError(f'motor_param_table has existing value, which should be '
                                         f'set to None after read by run_policy.')

                    p_table[param_value.index] = param_value

                elif ext_id.comm_type == CommunicationType.MOTOR_FEEDBACK:
                    motor_state_frame = RSProtocolParser.motor_state_feedback(data2=ext_id.data2, data=msg.data,
                                                                              ts=msg.timestamp)
                    await alogger.debug(f'{motor_state_frame}')

                    # depending on the run_policy process to fetch obs, so no need to use
                    # `await` to yield our cpu core, and even yield, this will not accelerate
                    # the run_policy process on another cpu core.
                    # NOTE: if the deque is full, the left most element will be dropped.
                    if (len(self.motor_state_q[motor_state_frame.can_id]) >=
                            self.motor_state_q[motor_state_frame.can_id].maxlen):
                        raise ValueError(f'motor_state_q_dict is full. check the run_policy fetch freq.')

                    self.motor_state_q[motor_state_frame.can_id].append(motor_state_frame)

                elif ext_id.comm_type == CommunicationType.GET_DEVICE_ID:
                    mcu_id = RSProtocolParser.motor_device_id(data2=ext_id.data2, data=msg.data)
                    await alogger.debug(f'{mcu_id}')

                else:
                    await alogger.warning(f'not supported rcv msg comm type: {ext_id.comm_type}')

            except (ValueError, struct.error) as exc:
                await alogger.error(f'unpack/decode msg error: {exc=:} {type(exc)=:} . check the msg data: {msg}')
                # not raise.

            except Exception as exc:
                await alogger.error(f'parse rcv msg task exception: {exc=:} {type(exc)=:} ')
                # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task-group.
                raise exc

            finally:
                self._rcv_buffer_q.task_done()

    async def _produce_send_msg(self):
        try:
            while True:
                tx_msg = \
                RSProtocolBuilder.write_single_param(motor_can_id=np.asarray(self._motor_can_id, dtype=np.uint32),
                                                     host_can_id=self.host_can_id,
                                                     index=index,
                                                     param_value=[value],
                                                     param_spec=param_spec
                                                     )[0]
                await alogger.debug(f'write single param can msg: {tx_msg}')
                await self._send_buffer_q.put(tx_msg)

        except Exception as exc:
                # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task_group.
                await alogger.error(f'produce send msg task failed: {exc=:} {type(exc)=:}')
                raise exc

        finally:
            pass


    async def _connect(self):
        assert self.bus is None, "Client is already started."

        # NOTE: `sudo ip link set can0 up type can bitrate 1000000` first.
        try:
            filters = [
                # 29-bit mask.
                {'can_id': self.host_can_id, 'can_mask': 0xff, 'extended': True},
            ]
            self.bus = SocketcanBus(channel=self.channel_name,
                                    can_filters=filters)
        except Exception as exc:
            await alogger.error(f'create socket bus failed: channel: {self.channel_name} {exc=:} {type(exc)=:}')
            raise exc

    def _disconnect(self):
        """Disconnects from the RobStride motors."""

        # Ensure motors are disabled at the end.
        # using block io send.
        self.set_torque_enabled(motor_ids=self._motor_can_id, enabled=False)

        self.bus.shutdown()
        self.bus = None

        # self.motor_state_queue_dict.clear()
        # self.motor_param_queue_dict.clear()

        if self in RobStrideClient.OPEN_CLIENTS:
            RobStrideClient.OPEN_CLIENTS.remove(self)

    async def main_job(self):
        """Connects to the motors, and start send / rcv msgs loop.

        NOTE: This should be called after all RobstrideClients on the same
            process are created.
        """

        await self._connect()

        # Start with all motors enabled.  NO, I want to set settings before enabled
        # self.set_torque_enabled(self._motor_ids, True)

        file_dsc: int = -1
        try:
            file_dsc = self.bus.fileno()
        except NotImplementedError as exc:
            # Bus doesn't support fileno, we fall back to thread based reader
            await  alogger.error(f'bus not support fileno, we can not use it for async read/write.')
            raise exc

        loop = asyncio.get_running_loop()
        if loop is not None and file_dsc >= 0:
            # Use bus file descriptor to watch for messages
            loop.add_reader(file_dsc, self._on_read_available )
            loop.add_writer(file_dsc, self._on_write_available )

        try:
            async with asyncio.TaskGroup() as tg:
                task_rcv = tg.create_task(self._parse_rcv_msg())
                task_snd = tg.create_task(self._produce_send_msg())

        except Exception as exc:
            await alogger.error(f'run task group failed: {exc=:} {type(exc)=:} ' )

        finally:
            await aprint(f'exit main job...')

            # clear loop reader/writer callback.
            loop.remove_reader(file_dsc)
            await self._rcv_buffer_q.join()
            await aprint(f'rcv buffer cleared.')

            # stop tx_xxx_msg ...
            await self._send_buffer_q.join()
            loop.remove_writer(file_dsc)
            await aprint(f'snd buffer cleared.')

            # TODO:
            # flush buffer q:
            # SND_BUFFER_Q.shutdown()
            # RCV_BUFFER_Q.shutdown()

            # task_rcv.result()...
            self._disconnect()

    def set_torque_limit(
        self,
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
        assert len(self._motor_can_id) == len(max_torque)
        assert np.all(max_torque > 0)

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


    def _read_table_single_value_helper(self,*, name: TableValueName, into_cache: bool, retries: int = 0) \
            -> Tuple[float,npt.NDArray[Any]]:
        comm_time, value_arr_list = self._sync_read_helper(_TableValueReadSpec[name])
        assert len(value_arr_list)==1 and len(value_arr_list)==len(_TableValueReadSpec[name].result_dtype)
        if into_cache:
            assert self._cached_read_data_dict[name].dtype == _TableValueReadSpec[name].result_dtype[0]
            self._cached_read_data_dict[name] = value_arr_list[0].copy()

        # NOTE: if the ID does not return pkt, the corresponding value is `np.nan`.
        return comm_time, value_arr_list[0]

    def read_model_number_lazy(self, wait_sec:float) -> npt.NDArray[np.uint16]:
        _, ret = self._read_table_single_value_helper(name=TableValueName.model, into_cache=False)
        return ret

    # non block.
    def read_pos_nowait(self, retries: int = 0) -> Tuple[float,npt.NDArray[np.float32]]:
        # return self._read_table_single_value_helper(name=TableValueName.pos, into_cache=True)

        if self.motor_state_q[_id] is empty, raise error? or block?

        return [ self.motor_state_q[_id].popleft().pos
                 for _id in self._motor_can_id ]

    def read_vel_nowait(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        return self._read_table_single_value_helper(name=TableValueName.vel, into_cache=True)

    # executed in run_policy process. not real time data, we can set wait time.
    def read_voltage_lazy(self, retries: int = 0, wait_sec:float) -> Tuple[float, npt.NDArray[np.float16]]:
        motor_v = []
        index = RS_param_table_spec['VBUS'].index

        # build read msg.

        # put read msg into send deque.

        # block wait???

        # read from table.

        for _id in self._motor_can_id:
            p_table = self.motor_param_table[_id]
            voltage = p_table[index]
            assert voltage is not None
            motor_v.append(voltage)
            p_table[index] = None

        return motor_v


    def set_goal_pos(self, pos: npt.NDArray[np.float32]):
        """Writes the given desired positions.

        Args:
            pos: The joint angles in radians to write. in rad of single turn.signed value,
             to represent rotor direction.
             element order in `pos` must be same as self.motor_can_id.
        """
        assert len(self._motor_can_id) == len(pos)
        # TODO: only allow -2Pi ~ 2Pi.
        if not np.all(np.abs(pos) < 2 * np.pi):
            raise ValueError(f'not allowed goal pos: {pos}, which should be in [-2pi, 2pi] ')

        self._sync_write_impl(param=[signed_to_proto_param_bytes_v2(value=int(_v), size=size)
                                          for _v in steps],  # addr 42, 43,
                              address=SMS_STS_SRAM_Table_RW.GOAL_POSITION_L,
                              write_ids=motor_ids)

    def set_goal_accel(self, accel: npt.NDArray[np.float32]):
            """Writes the given accel.

            Args:
                accel:
            """
            assert len(self._motor_can_id) == len(accel)

            self._sync_write_impl(param=[int(_v).to_bytes(length=size, byteorder='little', signed=False)
                                         for _v in steps],  # addr 41,
                                  address=SMS_STS_SRAM_Table_RW.GOAL_ACCEL,
                                  write_ids=motor_ids)

    def set_goal_vel(self, vel: npt.NDArray[np.float32]):
        """Writes the given vel.

        Args:
            vel:
        """
        assert len(self._motor_can_id) == len(vel)

        self._sync_write_impl(param=[signed_to_proto_param_bytes_v2(value=int(_v), size=size)
                                     for _v in steps],  # addr 46,
                              address=SMS_STS_SRAM_Table_RW.GOAL_VEL_L,
                              write_ids=motor_ids)


    def set_control_mode(self, value: int|Sequence[int]):
        # TODO: only allow 0,1,2,3
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.CONTROL_MODE,value=value)

    def set_kp(self, value: int | Sequence[int]):
        #TODO: not allow 0 value for kp....
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.KP, value=value)

    def set_kp_kd_ki(self, *, kp: Sequence[int], kd: Sequence[int], ki: Sequence[int]):
        # kp:21, kd:22 , ki:23 consecutive address.
        self._set_multiple_bytes_param_helper(address=SMS_STS_EEPROM_Table_RW.KP,
                                              value=list(zip(kp, kd, ki)) )

    def set_mech_zero(self):
        pass



    # def __enter__(self):
    #     """Enables use as a context manager."""
    #     self.start()
    #     return self
    #
    # def __exit__(self, *args):
    #     """Enables use as a context manager."""
    #     self.stop()
    #
    # def __del__(self):
    #     """Automatically disconnect on destruction."""
    #     self.stop()


def _client_cleanup_handler():
    """Handles cleanup of open Feite clients by forcibly closing active connections.

    Iterates over all open Feite clients and checks if their port handlers are in use.
    If a port handler is active, logs a warning message and forces the client to close
    by setting the port handler's `is_using` attribute to False and disconnecting the client.
    """
    open_clients: List[RobStrideClient] = list(RobStrideClient.OPEN_CLIENTS)  # type: ignore
    for open_client in open_clients:
        open_client._disconnect()

# Register global cleanup function.
atexit.register(_client_cleanup_handler)
