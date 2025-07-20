import atexit
import sys
import time
import subprocess
import multiprocessing as mp
from typing import (Any, Dict, List, Sequence,
                    Set, ClassVar, OrderedDict)
# deque does not use lock, but its append/popleft is atomic operation.
from collections import OrderedDict, deque
import asyncio
from queue import Full
from aiologger import Logger
from aiologger.levels import LogLevel
from aioconsole import aprint
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import struct
import can
from can.interfaces.socketcan import SocketcanBus
from enum import Enum,auto

from toddlerbot.actuation.robstride_sdk import *
from toddlerbot.actuation._module_logger import logger

alogger = Logger.with_default_handlers(level=LogLevel.DEBUG)

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

class RSBaudRate(Enum):
    BPS_1M = auto()
    BPS_500K = auto()
    BPS_250K = auto()
    BPS_125K = auto()

    def convert_to_rs_cmd(self)->int:
        if self == RSBaudRate.BPS_1M:
            return BaudRateCmd.BPS_1M
        elif self == RSBaudRate.BPS_500K:
            return BaudRateCmd.BPS_500K
        elif self == RSBaudRate.BPS_250K:
            return BaudRateCmd.BPS_250K
        else:
            return BaudRateCmd.BPS_125K

    def convert_to_value(self)->int:
        if self == RSBaudRate.BPS_1M:
            return 1_000_000
        elif self == RSBaudRate.BPS_500K:
            return 500_000
        elif self == RSBaudRate.BPS_250K:
            return 250_000
        else:
            return 125_000

# param table index: 0x7005
class RSRunMode(Enum):
    MOTION = auto()  # 运控模式
    PP_POSITION = auto()  # PP位置模式
    SPEED = auto()  # 速度模式
    CURRENT = auto()  # 电流模式
    CSP_POSITION = auto()  # CSP位置模式

    def convert_to_rs_cmd(self) -> int:
        if self == RSRunMode.MOTION:
            return RunModeCmd.MOTION

        elif self == RSRunMode.PP_POSITION:
            return RunModeCmd.PP_POSITION

        elif self == RSRunMode.SPEED:
            return RunModeCmd.SPEED

        elif self == RSRunMode.CURRENT:
            return RunModeCmd.CURRENT
        else:
            return RunModeCmd.CSP_POSITION


class RSReportPeriod(Enum):
    P_10MS:auto()
    P_15MS: auto()
    P_20MS: auto()
    P_25MS: auto()
    P_30MS: auto()
    P_35MS: auto()
    P_40MS: auto()
    P_45MS: auto()

    def convert_to_rs_cmd(self) -> int:
        if self == RSReportPeriod.P_10MS:
            return ReportPeriodCmd.P_10MS

        elif self == RSReportPeriod.P_15MS:
            return ReportPeriodCmd.P_15MS

        elif self == RSReportPeriod.P_20MS:
            return ReportPeriodCmd.P_20MS

        elif self == RSReportPeriod.P_25MS:
            return ReportPeriodCmd.P_25MS

        elif self == RSReportPeriod.P_30MS:
            return ReportPeriodCmd.P_30MS

        elif self == RSReportPeriod.P_35MS:
            return ReportPeriodCmd.P_35MS

        elif self == RSReportPeriod.P_40MS:
            return ReportPeriodCmd.P_40MS

        else:
            return ReportPeriodCmd.P_45MS


# bring up the can interface:
def _bring_up_can_interface(if_name:str, bitrate: int):
    os_type = sys.platform.casefold()
    try:
        if os_type != "linux":
            raise NotImplementedError

        if_status:str = subprocess.run(f'ip link show {if_name}',shell=True,
                                       capture_output=True, text=True, check=True).stdout.strip().casefold()
        if 'state up' in if_status:
            logger.warning(f'{if_name} is already UP.')

        else:
            sh_cmd = f"sudo ip link set {if_name} up type can bitrate {bitrate}"
            logger.info(f"bring up can interface cmd: {sh_cmd}")

            result = subprocess.run(
                sh_cmd, shell=True, text=True,
                check=True, stdout=subprocess.PIPE,
            )
            if ret_code:=result.returncode != 0:
                logger.error(f"sh cmd execute error, cmd:{sh_cmd},"
                             f"return code: {ret_code}, "
                             f"result stdout: {result.stdout.strip()}, "
                             f"result stderr:{result.stderr.strip()}, ")
                raise OSError(f'bring up can interface sh cmd executed error: {sh_cmd}')

    except Exception as exc:
        logger.error(f'bring up can interface failed: {exc=:} {type(exc)=:} ')
        raise exc

    finally:
        # blocking io
        time.sleep(0.1)

# shutdown the can interface:
def _shutdown_can_interface(if_name:str):
    os_type = sys.platform.casefold()
    try:
        if os_type != "linux":
            raise NotImplementedError
        else:
            sh_cmd = f"sudo ip link set {if_name} down "

        logger.warning(f"shutdown can interface ---> shell cmd: {sh_cmd}")

        result = subprocess.run(
            sh_cmd, shell=True, text=True, check=True, stdout=subprocess.PIPE,
        )
        if ret_code := result.returncode != 0:
            logger.error(f"sh cmd execute error, cmd:{sh_cmd},"
                                f"return code: {ret_code}, "
                                f"result stdout: {result.stdout.strip()}, "
                                f"result stderr:{result.stderr.strip()}, ")
            raise OSError(f'shutdown can interface sh cmd executed error: {sh_cmd}')

    except Exception as exc:
        logger.error(f'shutdown can interface failed: {exc=:} {type(exc)=:} ')
        raise exc

    finally:
        time.sleep(0.1)



class RobStrideIOProc:
    """Client for communicating with a group of Feite motors.
     should use individual client for one can bus.
    NOTE: only supports can ExtID.
    """

    # The currently open clients. class variable.
    OPEN_CLIENTS: ClassVar[Set[Any] ] = set()

    # instance variable.
    bus: SocketcanBus | None

    """
    send data to motor:
    API -> build can msg -> put into _msg_to_send_buffer_q -> put into _send_buffer_q when
    select_io_writable is ready  -> send by bus immediately.
    
    rcv data from motor:
    put into _rcv_buffer_q when select_io_readable is ready -> parse can msg -> put motor data
    into _motor_state_q and _motor_param_table  -> get data through API, and clear cached data.
    
    """

    # used by loop.reader/writers.
    _loop_send_buffer_q: asyncio.Queue[can.Message]
    _loop_rcv_buffer_q: asyncio.Queue[can.Message]

    # used by motor operation API, cache msg to be sent out.
    # _app_msg_send_buffer_q: deque[can.Message]
    # _app_msg_send_buffer_q: asyncio.Queue[can.Message]

    # process/thread safe queue.
    _app_msg_send_buffer_q: mp.Queue  #[can.Message]

    # index by motor can id.
    # TODO: protect by lock?
    _motor_state_q: OrderedDict[int, deque[MotorStateFrame]]
    _motor_param_table: OrderedDict[int, Dict[int, SingleParamValue | None]]

    def __init__(
        self,*,
        motor_can_id: Sequence[int],         # ids of a group of actuators.
        host_can_id: int,                    # 0xfe
        channel: str,                   #= "can0",
        baud_rate: RSBaudRate,            # = 1_000_000 default for RS
        # init_target_pos: npt.NDArray[np.float32]|None,  # set to `init_pos` in config.json, or None for calibrate-zero.
        # lazy_connect: bool,                #= False,
        # rcv_timeout_ms: int,               #= 5,    #usb serial latency timer, default 5 ms.
    ):
        """Initializes a new client.
        Args:
            motor_can_id:
            host_can_id:
            channel:
            baud_rate:
            # init_target_pos:
            # lazy_connect:
            # rcv_timeout_ms:
           """

        # NOTE: _motor_ids is not guaranteed to be consecutive, i.e, could be [44, 1, 230,...]. so we
        # could not use id as array index directly.
        self._motor_can_id:npt.NDArray[np.uint32] = np.asarray(motor_can_id,dtype=np.uint32)  # not changed after instantiating a FeiteClient instance.

        assert  np.all(0 < self._motor_can_id) and np.all( self._motor_can_id <= 0x7f )

        assert 0x7f < host_can_id <= 0xfe
        self.host_can_id = host_can_id

        self.channel = channel

        self.baud_rate = baud_rate
        # self._init_target_pos = init_target_pos

        # self.lazy_connect = lazy_connect
        # self.rcv_timeout_ms = rcv_timeout_ms
        self.bus = None

        # self._app_msg_send_buffer_q = deque(maxlen= 5*len(motor_can_id))
        # self._app_msg_send_buffer_q = asyncio.Queue(maxsize= 5*len(motor_can_id))

        # process/thread safe queue.
        self._app_msg_send_buffer_q = mp.Queue(maxsize= 5*len(motor_can_id))

        # index through motor can id.
        # TODO: adjust queue size according to control freq.
        # we set the deque size to 1, to make the `obs` always be the latest one.
        self._motor_state_q = OrderedDict((_id, deque(maxlen=1)) for _id in motor_can_id)
        self._motor_param_table = OrderedDict((_id, dict()) for _id in motor_can_id)

        # TODO: protected by lock.
        self._loop_rcv_buffer_q: asyncio.Queue[can.Message] = asyncio.Queue(maxsize=10 * len(motor_can_id))
        self._loop_send_buffer_q: asyncio.Queue[can.Message] = asyncio.Queue(maxsize=10 * len(motor_can_id))

        self._connect()
        RobStrideIOProc.OPEN_CLIENTS.add(self)


    # NOTE: callback invoked in running_loop: do not implement as co-routine directly:
    def _on_read_available(self) -> None:
        if msg := self.bus.recv(timeout=0):
            # if buffer q is full, raise error directly.
            try:
                # append/popleft op is atomic, no need to add lock.
                self._loop_rcv_buffer_q.put_nowait(msg)
            except Exception as exc:
                print(f'put rcv msg to buffer failed: {exc=:},  {type(exc)=:}. maybe increase buffer size.')
                raise

    # NOTE: callback invoked in running_loop: do not implement as co-routine directly:
    def _on_write_available(self) -> None:
        if not self._loop_send_buffer_q.empty():
            try:
                msg = self._loop_send_buffer_q.get_nowait()
                self.bus.send(msg, timeout=0)
                self._loop_send_buffer_q.task_done()

            except Exception as exc:
                raise OSError(f'send msg from buffer queue failed: {exc=:} {type(exc)=:}')

    async def _parse_rcv_msg(self) -> None:
        while True:
            msg: can.Message|None = None
            try:
                msg = await self._loop_rcv_buffer_q.get()

                await alogger.debug(f'rcv msg: {msg}')

                ext_id = RSProtocolParser.decode_ext_id(msg.arbitration_id)

                # filtered by socket can.
                # if ext_id.dest_can_id != self.host_can_id:
                #     raise ValueError(f'rcv msg dest can id is not host, dest id: {ext_id.dest_can_id}')

                await alogger.debug(f'parse msg result--->')
                if ext_id.comm_type == CommunicationType.SINGLE_PARAM_READ:
                    param_value = RSProtocolParser.single_param(data2=ext_id.data2, data=msg.data, ts=msg.timestamp)
                    await alogger.debug(f'{param_value}')

                    p_table: Dict[int, SingleParamValue] = self._motor_param_table[param_value.can_id]

                    if param_value.index in p_table and p_table[param_value.index] is not None:
                        raise ValueError(f'motor_param_table has existing value, which should be '
                                         f'set to None after read by run_policy.')

                    p_table[param_value.index] = param_value

                elif ext_id.comm_type == CommunicationType.MOTOR_FEEDBACK:
                    # TODO: add error handler.
                    motor_state_frame = RSProtocolParser.motor_state_feedback(data2=ext_id.data2, data=msg.data,
                                                                              ts=msg.timestamp)
                    await alogger.debug(f'{motor_state_frame}')

                    # depending on the run_policy process to fetch obs, so no need to use
                    # `await` to yield our cpu core, and even yield, this will not accelerate
                    # the run_policy process on another cpu core.
                    # NOTE: if the deque is full, the left most element will be dropped.
                    if (len(self._motor_state_q[motor_state_frame.can_id]) >=
                            self._motor_state_q[motor_state_frame.can_id].maxlen):
                        raise ValueError(f'motor_state_q_dict is full. check the run_policy fetch freq.')

                    self._motor_state_q[motor_state_frame.can_id].append(motor_state_frame)

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
                if msg is not None:
                    self._loop_rcv_buffer_q.task_done()

    async def _dump_send_msg(self):
        while True:
            # an optimized way to yield.
            await asyncio.sleep(0.)
            # 1 ns.
            # await asyncio.sleep(1e-9)

            if not self._app_msg_send_buffer_q.empty():
                try:
                    tx_msg = self._app_msg_send_buffer_q.get_nowait()   #non-block.
                    await alogger.debug(f'dump send msg: {tx_msg}')
                    await self._loop_send_buffer_q.put(tx_msg)

                except Exception as exc:
                        # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task_group.
                        await alogger.error(f'dump send msg task failed: {exc=:} {type(exc)=:}')
                        raise exc

    # async def _dump_send_msg(self):
    #     while True:
    #         tx_msg: can.Message|None = None
    #         try:
    #             tx_msg = await self._app_msg_send_buffer_q.get()
    #             await alogger.debug(f'write single param can msg: {tx_msg}')
    #             await self._loop_send_buffer_q.put(tx_msg)
    #
    #         except Exception as exc:
    #                 # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task_group.
    #                 await alogger.error(f'dump send msg task failed: {exc=:} {type(exc)=:}')
    #                 raise exc
    #
    #         finally:
    #             if tx_msg is not None:
    #                 self._app_msg_send_buffer_q.task_done()

    # async def _dump_send_msg(self):
    #     try:
    #         while True:
    #             await asyncio.sleep(0.) ---> an optimized way to yield.
    #             while self._app_msg_send_buffer_q:
    #                 tx_msg = self._app_msg_send_buffer_q.popleft()
    #                 await alogger.debug(f'write single param can msg: {tx_msg}')
    #                 await self._loop_send_buffer_q.put(tx_msg)
    #
    #     except Exception as exc:
    #             # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task_group.
    #             await alogger.error(f'produce send msg task failed: {exc=:} {type(exc)=:}')
    #             raise exc
    #
    #     finally:
    #         pass

    def _connect(self):
        assert self.bus is None, "Client is already started."

        # NOTE: `sudo ip link set can0 up type can bitrate 1000000` first.
        # TODO: modify Ubuntu system file to bring up can0 automatically.
        bitrate:int = self.baud_rate.convert_to_value()
        _bring_up_can_interface(self.channel, bitrate=bitrate)

        try:
            filters = [
                # 29-bit mask.
                {'can_id': self.host_can_id, 'can_mask': 0xff, 'extended': True},
            ]
            self.bus = SocketcanBus(channel=self.channel,
                                    can_filters=filters)
        except Exception as exc:
            alogger.error(f'create socket bus failed: channel: {self.channel} {exc=:} {type(exc)=:}')
            raise exc

    def disconnect(self):
        """Disconnects from the RobStride motors."""

        logger.warning(f'disconnect from RS motors--->')

        if self.bus is None:
            # already disconnected.
            return

        # Ensure motors are disabled at the end.
        # using block-io to send, asyncio task is already shutdown.
        motor_disable_msg: List[can.Message] = RSProtocolBuilder.motor_disable(motor_can_id=self._motor_can_id,
                                                                               host_can_id=self.host_can_id)
        for _m in motor_disable_msg:
            # TODO: handle timeout exception.
            self.bus.send(_m,0.1)

        time.sleep(0.5)
        self.bus.shutdown()
        self.bus = None

        # self.motor_state_queue_dict.clear()
        # self.motor_param_queue_dict.clear()

        # if self in RobStrideClient.OPEN_CLIENTS:
        #     RobStrideClient.OPEN_CLIENTS.remove(self)

        _shutdown_can_interface(self.channel)
        time.sleep(0.5)

    async def send_rcv_task(self):
        """Connects to the motors, and start send / rcv msgs loop.

        NOTE: This should be called after all RobstrideClients on the same
            process are created.
        """

        # await self._connect()

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
                task_snd = tg.create_task(self._dump_send_msg())

        except Exception as exc:
            await alogger.error(f'run task group failed: {exc=:} {type(exc)=:} ' )

        finally:
            await alogger.warning(f'---> exit RobStrideClient send_rcv_task...')

            # clear loop reader/writer callback.
            loop.remove_reader(file_dsc)

            # await self._loop_rcv_buffer_q.join()
            # await aprint(f'rcv buffer cleared.')


            # flush _app_msg_send_buffer_q
            # await self._app_msg_send_buffer_q.join()

            # _dump_send_msg task finished, no new msg dump to _loop_send_buffer_q.
            # await self._loop_send_buffer_q.join()
            loop.remove_writer(file_dsc)

            # await aprint(f'snd buffer cleared.')

            # task_rcv.result()...
            # await self.disconnect()



    # RS manual suggest not modify torque_limit and other protection mode parameter.
    # def set_torque_limit(
    #         self,
    #         max_torque: npt.NDArray[np.float32],
    #         retries: int = -1,
    #         retry_interval: float = 0.25,)->None:
    #     pass

    # def __enter__(self):
    #     """Enables use as a context manager."""
    #     self.start()
    #     return self
    #
    # def __exit__(self, *args):
    #     """Enables use as a context manager."""
    #     self.stop()
    #

    def __del__(self):
        """Automatically disconnect on destruction."""
        # allow call on a client which is already disconnected explicitly.
        logger.warning(f' called on RobStrideClient.__del__(), disconnect RS motors:')
        self.disconnect()
        if self in RobStrideIOProc.OPEN_CLIENTS:
            RobStrideIOProc.OPEN_CLIENTS.remove(self)




def _client_cleanup_handler():
    """Handles cleanup of open RS clients by forcibly closing active connections.

    Iterates over all open Feite clients and checks if their port handlers are in use.
    If a port handler is active, logs a warning message and forces the client to close
    by setting the port handler's `is_using` attribute to False and disconnecting the client.
    """

    logger.warning(f' called on _client_cleanup_handler, disconnect RS motors:')

    open_clients: List[RobStrideIOProc] = list(RobStrideIOProc.OPEN_CLIENTS)  # type: ignore
    for open_client in open_clients:
        # TODO: how to finish the main_job asyncio task group?
        open_client.disconnect()

# Register global cleanup function.
atexit.register(_client_cleanup_handler)


if __name__ == '__main__':
    # alogger = Logger.with_default_handlers(level=LogLevel.DEBUG)
    client = RobStrideIOProc(motor_can_id=[0x7f],
                             host_can_id=0xfe,
                             channel='can0',
                             baud_rate=RSBaudRate.BPS_1M,
                             )
    asyncio.run(client.send_rcv_task())
    asyncio.run(alogger.shutdown())