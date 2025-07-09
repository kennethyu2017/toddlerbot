from typing import Set,Tuple, Type, Callable
import numpy as np
import asyncio
from aioconsole import ainput, aprint
from aiologger import Logger

from can import Message
from can.interfaces.socketcan import *

from toddlerbot.actuation.robstride_sdk import *

alogger = Logger.with_default_handlers()

CAN_CHANNEL_NAME : str = r'can0'    #r'PCAN_USBBUS1'
MOTOR_CAN_ID_SET: Set[int] = {0}
HOST_CAN_ID: int = 0xfe

# LISTENERS: List[Callable[[Message], None]] = []
# TALKERS: List[Callable[[Message], None]] = []

# TODO: protected by lock.
RCV_BUFFER_Q: asyncio.Queue[Message] | None = asyncio.Queue(maxsize=50)
SND_BUFFER_Q: asyncio.Queue[Message] | None = asyncio.Queue(maxsize=30)
# RCV_BUFFER_Q: Deque[Message] | None = deque(maxlen=50)
# SND_BUFFER_Q: Deque[Message] | None = deque(maxlen=30)

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

        except Exception as exc:
            raise OSError(f'send msg from buffer queue failed: {exc=:} {type(exc)=:}')


async def _tx_write_param_msg(*, value: int|float,
                              motor_can_id: int, index: int):

    tx_msg = RSProtocolBuilder.write_single_param(motor_can_id: npt.NDArray[np.uint32],
    host_can_id: int,
                           index: int,
                           param_value: List[float|int],
                           param_n_bytes:int,
                           param_dtype:Type,
                           param_signed:bool)

    assert len(txpkt) == length

    await aprint(f'--- Write [ID: {motor_id:>2d}] addr:[ {addr:>2d}] length:[{length:>2d}]  txpkt --->')
    for _b in txpkt:
        await aprint(f'0x{_b:02x}')

    await aprint(f'--- end of txpkt.')

    # comm_result, error = writer.writeTxRx(scs_id=motor_id,
    #                                       address=addr,
    #                                       length=len(txpkt),
    #                                       data=txpkt)
    #
    # if comm_result != CommResult.SUCCESS:
    #     # raise IOError(f'writer.writeTxRx comm error : {writer.getTxRxResult(comm_result)}')
    #     await aprint(f'Warning: writer.writeTxRx comm error : {writer.getTxRxResult(comm_result)}, pls check the motor ID / motor connection.')
    #
    # if error != 0:
    #     raise ValueError(f'writer.writeTxRx got error from motor : {writer.getRxPacketError(error)} ')


async def _tx_read_param_msg(*, motor_can_id: int, index: int):

    tx_msg:Message = RSProtocolBuilder.read_single_param(motor_can_id=np.asarray([motor_can_id],dtype=np.uint32),
                                                 host_can_id=HOST_CAN_ID,
                                                 index=index)[0]
    await SND_BUFFER_Q.put(tx_msg)
    await alogger.debug(f'read single param can msg: {tx_msg}')

    # if rxpkt is None:
    #     await aprint(f'Warning: rxpkt is None, pls check the motor ID / motor connection.')
    #     return
    #
    # if comm_result != CommResult.SUCCESS:
    #     raise IOError(f'reader.readTxRx comm error:  {reader.getTxRxResult(comm_result)} ')
    #
    # else:
    #     await aprint(f'--- Read [ID: {motor_id:>2d}] addr:[ {addr:>2d}] length:[{length:>2d}] result rxpkt --->')
    #     for _b in rxpkt:
    #         await aprint(f'0x{_b:02x}')
    #     await aprint(f'--- end of rxpkt.')
    #
    #     if len(rxpkt) <= 2 :
    #         # highest bit represent sign.
    #         signed_dec_value:int = proto_param_bytes_to_signed_v2(param=rxpkt)
    #
    #         unsigned_dec_value:int = int.from_bytes(rxpkt, byteorder='little',signed=False)
    #         await aprint(f'+++ when length <=2 , we can parse rxpkt to signed decimal value: {signed_dec_value}, '
    #               f'unsigned decimal value:{unsigned_dec_value}')
    #
    # if error != 0:
    #     raise ValueError(f'reader.readTxRx got error from motor: {reader.getRxPacketError(error)} ')


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


async def _input_int_or_float_helper(*, dtype: Type[int, float],
                                    legal_check: Callable[[float|int], bool],
                                    prompt: str) -> float|int:
    _value = None

    while True:
        # await asyncio.sleep(1)
        try:
            _value = dtype(await ainput(prompt))

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
                _value = None
                await aprint(f'got illegal input: {_value}, should be in range: {legal_min_max} ')
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
    try:
        while True:
            # sleep 1ms
            # await asyncio.sleep(0.001)
            msg:Message = await RCV_BUFFER_Q.get()
            await alogger.debug(f'rcv msg: {msg}')

            ext_id = RSProtocolParser.decode_ext_id(msg.arbitration_id)

            if ext_id.dest_can_id != HOST_CAN_ID:
                raise ValueError(f'rcv msg dest can id is not host, dest id: {ext_id.dest_can_id}')

            await alogger.info(f'parse msg result--->')
            if ext_id.comm_type == CommunicationType.SINGLE_PARAM_READ:
                param_value = RSProtocolParser.single_param(data2=ext_id.data2, data=msg.data)
                await alogger.info(f'{param_value}')

            elif ext_id.comm_type == CommunicationType.MOTOR_FEEDBACK:
                motor_state_frame = RSProtocolParser.motor_state_feedback(data2=ext_id.data2, data=msg.data, ts=msg.timestamp)
                await alogger.info(f'{motor_state_frame}')

            elif ext_id.comm_type == CommunicationType.GET_DEVICE_ID:
                mcu_id = RSProtocolParser.motor_device_id(data2=ext_id.data2, data=msg.data)
                await alogger.info(f'{mcu_id}')

    except Exception as exc:
        await alogger.error(f'parse rcv data failed: {exc=:} {type(exc)=:} ')

    finally:
        await alogger.warning(f'exit rcv data parse...')


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
        raise

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
                await aprint(f'\n--- start READ motor param table ( only support read single motor and single param till now ):')
                motor_can_id = await _input_int_or_float_helper(dtype=int,
                                                                legal_check=lambda _x: _x in MOTOR_CAN_ID_SET,
                                                                prompt=f'\ninput read motor can id in choices {MOTOR_CAN_ID_SET} : ')

                index = await _input_int_or_float_helper(dtype=int,
                                                         legal_check=lambda _x:  0x7005 <= _x <= 0x7029,
                                                         prompt='\ninput read index [0x7005~0x7029] : ')

                await _tx_read_param_msg(motor_can_id=motor_can_id,index=index)

            elif read_or_write == 'w':
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
                    p_spec = param_table_spec[p_name]
                except KeyError as exc:
                    await alogger.error(f'key error: {exc=:} {type(exc)=:}')
                    raise

                value = await _input_int_or_float_helper(dtype=float,
                                                         legal_check=lambda _x: p_spec.min_max[0] <= _x <= p_spec.min_max[1],
                                                         prompt=f'\ninput write value in range {p_spec.min_max} : '
                                                         )

                await _tx_write_param_msg(value=value, motor_can_id=motor_can_id, index=index)

            else:
                # raise ValueError(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                await aprint(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                continue

    finally:
        await aprint(f'exiting...')
        # Close port
        bus.shutdown()

        # clear loop reader/writer callback.
        loop.remove_reader(file_dsc)
        loop.remove_writer(file_dsc)

        # TODO:
        # flush buffer q:

        # logger shutdown.
        await alogger.shutdown()

        await asyncio.sleep(1.0)


if __name__  == '__main__':
    try:
        asyncio.TaskGroup....
        asyncio.run(_interactive_cmd())
    finally:
        print(f'exiting...')
        # Close port
        bus.shutdown()

        # clear loop reader/writer callback.
        loop.remove_reader(file_dsc)
        loop.remove_writer(file_dsc)

        # TODO:
        # flush buffer q:

        # logger shutdown.
        await alogger.shutdown()

        await asyncio.sleep(1.0)
