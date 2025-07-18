import struct
from typing import Set, Type, Callable
import time
import numpy as np
import asyncio
from aioconsole import ainput, aprint
from aiologger import Logger
from can import Message
from can.interfaces.socketcan import SocketcanBus

from toddlerbot.actuation.robstride_sdk import *

alogger = Logger.with_default_handlers()

CAN_CHANNEL_NAME : str = r'can0'    #r'PCAN_USBBUS1'
MOTOR_CAN_ID_SET: Set[int] = {0x7f}  # default ID of RS.
HOST_CAN_ID: int = 0xfe

# LISTENERS: List[Callable[[Message], None]] = []
# TALKERS: List[Callable[[Message], None]] = []

# TODO: protected by lock.
RCV_BUFFER_Q: asyncio.Queue[Message] | None = asyncio.Queue(maxsize=50)
SND_BUFFER_Q: asyncio.Queue[Message] | None = asyncio.Queue(maxsize=30)

# for snd-rcv delay test.
_send_msg_time_stamp: float|None = None

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
# motor_data_dict = MotorData()

# NOTE: callback invoked in running_loop: do not implement as co-routine directly:
def _on_read_available(bus: SocketcanBus) -> None:
    if msg := bus.recv(timeout=0):
        # if buffer q is full, raise error directly.
        try:
            # append/popleft op is atomic, no need to add lock.
            RCV_BUFFER_Q.put_nowait(msg)
        except Exception as exc:
            print(f'put rcv msg to buffer failed: {exc=:},  {type(exc)=:}. maybe increase buffer size.')
            raise

# NOTE: callback invoked in running_loop: do not implement as co-routine directly:
def _on_write_available(bus: SocketcanBus) -> None:
    if not SND_BUFFER_Q.empty():
        try:
            msg = SND_BUFFER_Q.get_nowait()
            bus.send(msg, timeout=0)
            SND_BUFFER_Q.task_done()

        except Exception as exc:
            raise OSError(f'send msg from buffer queue failed: {exc=:} {type(exc)=:}')


async def _tx_write_param_msg(*, value: int|float,
                              motor_can_id: int,
                              index: int,
                              param_spec:ParamSpec):

    tx_msg = RSProtocolBuilder.write_single_param(motor_can_id=np.asarray([motor_can_id], dtype=np.uint32),
                                                  host_can_id=HOST_CAN_ID,
                                                  index=index,
                                                  param_value=[value],
                                                  param_spec=param_spec
                                                  )[0]

    await alogger.debug(f'write single param can msg: {tx_msg}')
    global _send_msg_time_stamp
    _send_msg_time_stamp = time.perf_counter()
    await SND_BUFFER_Q.put(tx_msg)


async def _tx_read_param_msg(*, motor_can_id: int,
                             index: int):

    tx_msg:Message = RSProtocolBuilder.read_single_param(motor_can_id=np.asarray([motor_can_id],dtype=np.uint32),
                                                 host_can_id=HOST_CAN_ID,
                                                 index=index)[0]

    await alogger.debug(f'tx read single param can msg: {tx_msg}')

    global _send_msg_time_stamp
    _send_msg_time_stamp = time.perf_counter()
    await SND_BUFFER_Q.put(tx_msg)


async def _input_str_value_helper(valid_input: Set[str], prompt: str) -> str:
    r_w: str = ''

    while not r_w in valid_input:
        try:
            r_w = await ainput(prompt)

        except ValueError as err:
            r_w = ''
            await aprint(f'key-in value error: {err}, {type(err)}')
            continue

        else:
            if r_w not in valid_input:
                await aprint(f'got illegal input: {r_w}, should be "r" or "w":  ')
                r_w =''

    return r_w


async def _input_int_or_float_helper(*, dtype: Type[int|float],
                                     legal_check: Callable[[float|int], bool],
                                     prompt: str
                                     ) -> float|int:
    _value = None

    while True:
        # await asyncio.sleep(1)
        try:
            raw_input:str = await ainput(prompt)
            if raw_input.casefold().startswith('0x'):
                _value = dtype(raw_input, base=16)
            else:
                _value = dtype(raw_input)

        except ValueError as err:
            # _value = value_range.stop
            # _value = illegal_sentinel
            await aprint(f'key-in value error: {err}, {type(err)}')
            await asyncio.sleep(.5)
            continue

        else:
            if legal_check(_value):
                break
            else:
                await aprint(f'got illegal input: {_value}, check the legal value range. ')
                _value = None
                await asyncio.sleep(.5)
                continue
                # _value = value_range.stop
                # _value = illegal_sentinel

    return _value


# async def _input_int_value_helper(*, legal_set: range | Set[int], illegal_sentinel: int, prompt: str) -> int:
#     # _value : int = value_range.stop # stop not in range.
#     _value = illegal_sentinel
#
#     while not _value in legal_set:
#         try:
#             _value = int(await ainput(prompt))
#
#         except ValueError as err:
#             # _value = value_range.stop
#             _value = illegal_sentinel
#             await aprint(f'key-in value error: {err}, {type(err)}')
#             continue
#
#         else:
#             if _value not in legal_set:
#                 await aprint(f'got illegal input: {_value}, should be in set: {legal_set} ')
#                 # _value = value_range.stop
#                 _value = illegal_sentinel
#
#     return _value


async def _parse_rcv_data()->None:
    while True:
        # sleep 1ms
        # await asyncio.sleep(0.001)

        try:
            msg: Message = await RCV_BUFFER_Q.get()

            global _send_msg_time_stamp
            if _send_msg_time_stamp is not None:
                snd_rcv_round_trip: float = time.perf_counter() - _send_msg_time_stamp
                # only count once.
                _send_msg_time_stamp = None
                await alogger.debug(f' send-rcv round trip: {snd_rcv_round_trip*1000.:.3f} ms')

            await alogger.debug(f'rcv msg: {msg}')

            ext_id = RSProtocolParser.decode_ext_id(msg.arbitration_id)

            if ext_id.dest_can_id != HOST_CAN_ID:
                raise ValueError(f'rcv msg dest can id is not host, dest id: {ext_id.dest_can_id}')

            await alogger.info(f'parse msg result--->')
            if ext_id.comm_type == CommunicationType.SINGLE_PARAM_READ:
                param_value = RSProtocolParser.single_param(data2=ext_id.data2, data=msg.data, ts=msg.timestamp)
                await alogger.info(f'{param_value}')

            elif ext_id.comm_type == CommunicationType.MOTOR_FEEDBACK:
                motor_state_frame = RSProtocolParser.motor_state_feedback(data2=ext_id.data2, data=msg.data, ts=msg.timestamp)
                await alogger.info(f'{motor_state_frame}')

            elif ext_id.comm_type == CommunicationType.GET_DEVICE_ID:
                mcu_id = RSProtocolParser.motor_device_id(data2=ext_id.data2, data=msg.data)
                await alogger.info(f'{mcu_id}')

            else:
                await alogger.warning(f'not supported rcv msg comm type: {ext_id.comm_type}')

        except (ValueError, struct.error) as exc:
            await alogger.error(f'unpack/decode msg error: {exc=:} {type(exc)=:} . check the msg data.')
            # not raise.

        except Exception as exc:
            await alogger.error(f'parse rcv data exception: {exc=:} {type(exc)=:} ')
            # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task-group.
            raise exc

        finally:
            RCV_BUFFER_Q.task_done()


async def _read_helper():
    await aprint(f'\n--- start READ motor param table ( only support read single motor and single param till now ):')
    motor_can_id = await _input_int_or_float_helper(dtype=int,
                                                    legal_check=lambda _x: _x in MOTOR_CAN_ID_SET,
                                                    prompt=f'\ninput read motor can id in choices {MOTOR_CAN_ID_SET} : ')

    index = await _input_int_or_float_helper(dtype=int,
                                             legal_check=lambda _x: 0x7005 <= _x <= 0x7029,
                                             prompt='\ninput read index [0x7005~0x7029] : ')

    await _tx_read_param_msg(motor_can_id=motor_can_id, index=index)


async def _write_helper():
    await aprint(f'\n--- start WRITE motor control table:')
    motor_can_id = await _input_int_or_float_helper(dtype=int,
                                                    legal_check=lambda _x: _x in MOTOR_CAN_ID_SET,
                                                    prompt=f'\ninput write motor id in choices {MOTOR_CAN_ID_SET} : ')

    index = await _input_int_or_float_helper(dtype=int,
                                             legal_check=lambda _x: 0x7005 <= _x <= 0x7029,
                                             prompt='\ninput read index [0x7005~0x7029] : ')

    # TODO: value validation should refer to readSpec min_max...
    try:
        p_name = param_table_index_to_name[index]
        p_spec = RS_param_table_spec[p_name]
    except KeyError as exc:
        await alogger.error(f'key error: {exc=:} {type(exc)=:}')
        raise exc

    value = await _input_int_or_float_helper(dtype=p_spec.dtype,
                                             legal_check=lambda _x: p_spec.min_max[0] <= _x <= p_spec.min_max[1],
                                             prompt=f'\ninput write value in range {p_spec.min_max} : '
                                             )

    await _tx_write_param_msg(value=value, motor_can_id=motor_can_id, index=index, param_spec=p_spec)


# NOTE: must propagate any exception up, to guarantee the structural task_group cancel all the remaining tasks
# inside same task_group.
async def _interactive_console()->None:

    # _set_usb_com_latency_timer(port_name=URT_1_DEV_NAME, latency_ms=5)
    try:
        # NOTE:set bit rate using cmd line.
        bus = SocketcanBus(channel=CAN_CHANNEL_NAME)
    except Exception as exc:
        await alogger.error(f'create socket bus failed: {exc=:} {type(exc)=:}')
        raise exc

    file_dsc: int = -1
    try:
        file_dsc = bus.fileno()
    except NotImplementedError as exc:
        # Bus doesn't support fileno, we fall back to thread based reader
        alogger.error(f'bus not support fileno, we can not use it for async read/write.')
        raise exc

    loop = asyncio.get_running_loop()
    if loop is not None and file_dsc >= 0:
        # Use bus file descriptor to watch for messages
        loop.add_reader(file_dsc, _on_read_available, bus)
        loop.add_writer(file_dsc, _on_write_available, bus)

    read_or_write : str = ''
    motor_can_id: int = -1
    index: int = -1

    try:
        while True:
            await asyncio.sleep(1.)

            read_or_write = await _input_str_value_helper( {'r', 'w'},
                                                           f'\nread or write? [r/w] : ')

            if read_or_write == 'r':
                await _read_helper()

            elif read_or_write == 'w':
                await _write_helper()

            else:
                # raise ValueError(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                await aprint(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                continue

    except Exception as exc:
        # NOTE: must propagate to up layer task_group to cancel the remaining tasks in task_group.
        await alogger.error(f'interactive console task failed: {exc=:} {type(exc)=:}')
        raise exc

    finally:
        await aprint(f'exit interactive console...')

        # clear loop reader/writer callback.
        loop.remove_reader(file_dsc)
        await RCV_BUFFER_Q.join()
        await aprint(f'rcv buffer cleared.')

        # stop tx_xxx_msg ...
        await SND_BUFFER_Q.join()
        loop.remove_writer(file_dsc)
        await aprint(f'snd buffer cleared.')

        # TODO:
        # flush buffer q:
        # SND_BUFFER_Q.shutdown()
        # RCV_BUFFER_Q.shutdown()

        # Close port
        # bus.flush_tx_buffer()
        bus.shutdown()

        # logger shutdown.
        await alogger.shutdown()

        await asyncio.sleep(.5)


async def _main():
    async with asyncio.TaskGroup() as tg:
        # input cmd and send msg onto bus.
        task_1 = tg.create_task(_interactive_console())

        # parse msg recv from bus.
        task_2 = tg.create_task(_parse_rcv_data())

    await aprint(f'all tasks are done : {task_1.result()=:} {task_2.result()=:}')


if __name__  == '__main__':
    asyncio.run(_main())

