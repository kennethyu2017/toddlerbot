"""Communication using the DynamixelSDK."""

##This is based off of the dynamixel SDK
import atexit
import time
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Dict, List, Optional, Sequence,Type,
                    Set, Tuple, ClassVar,Iterable,NamedTuple,Callable)
# from itertools import islice
# from functools import partial
import numpy as np
import numpy.typing as npt

from scservo_sdk import (PortHandler, ProtocolPacketHandler, GroupSyncReader, GroupSyncWriter,
                         CommResult, ByteOrder, SMS_STS_SRAM_Table_ReadOnly, SMS_STS_SRAM_Table_RW,
                         SMS_STS_EEPROM_Table_ReadOnly, SMS_STS_EEPROM_Table_RW, SMS_STS_Table_Data_Length)
from ._module_logger import logger


POS_RESOLUTION = 2.0 * np.pi / 4096  # 0.088 degrees per step.
VEL_RESOLUTION = 50 * POS_RESOLUTION  # 50 steps / second, 0.732 rpm.
VOLTAGE_RESOLUTION = 0.1
LOAD_PERCENTAGE_RESOLUTION = 0.1  # 0.1%
# See http://emanual.robotis.com/docs/en/dxl/x/xh430-v210/#goal-velocity
# DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
# DEFAULT_V_IN_SCALE = 0.1

class TableValueName(Enum):
    pos = 1
    vel = 2
    load = 3
    vin = 4
    model = 5
    pos_vel_load = 6  # together.

class PosVelLoadRecord(NamedTuple):
    comm_time: float
    pos: npt.NDArray[np.float32]
    vel: npt.NDArray[np.float32]
    load: npt.NDArray[np.float32]

def _feite_cleanup_handler():
    """Handles cleanup of open Feite clients by forcibly closing active connections.

    Iterates over all open Feite clients and checks if their port handlers are in use.
    If a port handler is active, logs a warning message and forces the client to close
    by setting the port handler's `is_using` attribute to False and disconnecting the client.
    """
    open_clients: List[FeiteClient] = list(FeiteClient.OPEN_CLIENTS)  # type: ignore
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logger.warning("Forcing client to close.")
        open_client.port_handler.is_using = False
        open_client.disconnect()



# # TODO: for feite protocol, the Two’s complement is not applied for the negative value. Use the BIT15 instead...
# def signed_to_proto_param_bytes_v1(value: int, size: int) -> bytes:
#     """Converts a signed integer to its unsigned equivalent based on the specified size.
#
#     Args:
#         value (int): The signed integer to convert.
#         size (int): The size in bytes of the integer type.
#
#     Returns:
#         int: The unsigned integer representation of the input value.the highest bit, e.g., BIT15, representing sign
#     """
#     # use the highest bit, e.g., BIT15, to represent sign.
#     param = None
#     if value < 0:
#         bit_size = 8 * size
#         highest_bit_1_value = 1 << (bit_size - 1)   # got unsigned int.
#         assert abs(value) < highest_bit_1_value
#         param = highest_bit_1_value + abs(value)
#     else:
#         param = value
#
#     return param.to_bytes(length=size, byteorder='little', signed=False)


# TODO: for feite protocol, the Two’s complement is not applied for the negative value. Use the BIT15 instead...
def _signed_to_proto_param_bytes_v2(*, value: int, size: int) -> bytes | bytearray:
    """Converts a signed integer to its unsigned equivalent based on the specified size.

    Args:
        value (int): The signed integer to convert.
        size (int): The size in bytes of the integer type.

    Returns:
        bytes: The unsigned integer representation of the input value.the highest bit, e.g., BIT15, representing sign
    """
    # use the highest bit, e.g., BIT15, to represent sign.
    abs_value = abs(value)
    param = abs_value.to_bytes(length=size, byteorder='little', signed=False)
    # [-32767, 32767] for 2 bytes.
    assert (param[-1] & 0x80 ) == 0
    # TODO: check the actuator real act.
    if value < 0:
        param = bytearray(param)
        param[-1] |= 0x80  # (1 << 7) got unsigned int.

    return param


# def proto_param_bytes_to_signed_v1(*, param:int, size: int) -> int:
#     """Converts an unsigned integer to a signed integer of a specified byte size.
#
#     Args:
#         param (int): The unsigned integer to convert. the highest bit, e.g., BIT15, representing sign.
#         size (int): The byte size of the integer.
#
#     Returns:
#         int: The signed integer representation of the input value.
#     """
#     bit_size = 8 * size
#     highest_bit_1_value = 1 << (bit_size - 1)  # got unsigned int.
#     # use the highest bit, e.g., BIT15, to represent sign.
#     if (param & highest_bit_1_value) != 0:
#         return  highest_bit_1_value - param
#     return param

def _proto_param_bytes_to_signed_v2(*, param: bytes | bytearray) -> int:
    """Converts an unsigned integer to a signed integer of a specified byte size.

    Args:
        param (bytes): The unsigned integer to convert. the highest bit, e.g., BIT15, representing sign.
        size (int): The byte size of the integer.

    Returns:
        int: The signed integer representation of the input value.
    """

    # use the highest bit, e.g., BIT15, to represent sign.
    # TODO: check the real act of actuator.
    sign:int = 1
    if (param[-1] & 0x80) !=0 :   # (1 << 7) got unsigned int.
        sign = -1
        if isinstance(param,bytes):
            param = bytearray(param)

        param[-1] &= 0x7f  # ~(1 << 7) got unsigned int.

    return sign * int.from_bytes(bytes=param, byteorder='little', signed=False)


def _parse_model(param: bytes| bytearray) -> int:
    assert len(param) == 2
    return int.from_bytes(bytes=param, byteorder='little', signed=False)

def _parse_pos(param: bytes | bytearray)->float:
    assert len(param) == 2
    # feedback param is present absolute steps in a single turn, w/o direction.
    step = int.from_bytes(bytes=param, byteorder='little', signed=False)
    assert 0 <= step <= 4095
    return step * POS_RESOLUTION  # rad in as single turn.

def _parse_vel(param: bytes | bytearray)->float:
    assert len(param) == 2
    # BIT15 is direction.
    vel = _proto_param_bytes_to_signed_v2(param=param)
    # vel resolution is 0.732 rpm, SM40BL max vel is 88 rpm.
    assert -120 <= vel <= 120
    return vel * VEL_RESOLUTION

def _parse_load(param: bytes | bytearray)->float:
    assert len(param) == 2
    # 0.1% of stall_torque.
    load = int.from_bytes(bytes=param, byteorder='little', signed=False)
    assert 0 <= load <= 1000
    return load * LOAD_PERCENTAGE_RESOLUTION


def _parse_vin(param: bytes | bytearray)->float:
    assert len(param) == 1
    vin = int.from_bytes(bytes=param, byteorder='little', signed=False)
    assert 0 <= vin < 140
    return vin * VOLTAGE_RESOLUTION


class _ReadSpec(NamedTuple):
    start_addr: int
    size: List[int]  # for multi-params read, e.g., `pos,vel,load` together.
    parser: List[ Callable[[bytes|bytearray], float|int] ]
    result_dtype: List[Type]


_TableValueReadSpec : Dict[TableValueName, _ReadSpec] = {
    TableValueName.pos: _ReadSpec(start_addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                     size=[SMS_STS_Table_Data_Length.PRESENT_POSITION],
                     parser=[_parse_pos],
                     result_dtype=[np.float32]),

    TableValueName.vel: _ReadSpec(start_addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VELOCITY_L,
                     size=[SMS_STS_Table_Data_Length.PRESENT_VELOCITY],
                     parser=[_parse_vel],
                     result_dtype=[np.float32]),

    TableValueName.load: _ReadSpec(start_addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_LOAD_L,
                      size=[SMS_STS_Table_Data_Length.PRESENT_LOAD],
                      parser=[_parse_load],
                      result_dtype=[np.float32]),

    TableValueName.vin: _ReadSpec(start_addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VOLTAGE,
                                     size=[SMS_STS_Table_Data_Length.PRESENT_VOLTAGE],
                                     parser=[_parse_vin],
                                     result_dtype=[np.float16]),

    TableValueName.model: _ReadSpec(start_addr=SMS_STS_EEPROM_Table_ReadOnly.MODEL_L,
                        size=[SMS_STS_Table_Data_Length.MODEL_NUMBER],
                        parser=[_parse_model],
                        result_dtype=[np.uint16]),

    # consecutive-address multiple values read.
    TableValueName.pos_vel_load: _ReadSpec(start_addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                        size=[SMS_STS_Table_Data_Length.PRESENT_POSITION,
                              SMS_STS_Table_Data_Length.PRESENT_VELOCITY,
                              SMS_STS_Table_Data_Length.PRESENT_LOAD],
                        parser=[_parse_pos, _parse_vel, _parse_load],
                        result_dtype=[np.float32, np.float32, np.float32])

}

@dataclass(init=False)
class FeiteGroupClient:
    """Client for communicating with a group of Feite motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients. class variable.
    OPEN_CLIENTS: ClassVar[Set[Any]] = set()

    # instance variable.
    port_handler: PortHandler

    def __init__(
        self,*,
        motor_ids: Sequence[int],  # ids of a group of actuators.
        port_name: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,  # default for SMS/STS series.
        lazy_connect: bool = False,
        latency_timer_ms: int = 1,    #usb serial latency timer, default 1ms.
    ):
        """Initializes a new client.

        Args:
        """

        # NOTE: _motor_ids is not guaranteed to be consecutive, i.e, could be [44, 1, 230,...]. so we
        # could not use id as array index directly.
        self._motor_ids = list(motor_ids)  # not changed after instantiating a FeiteClient instance.
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.lazy_connect = lazy_connect
        self.latency_timer_ms = latency_timer_ms

        self.port_handler = PortHandler(port_name=self.port_name,
                                        baud_rate=self.baud_rate,
                                        latency_timer=self.latency_timer_ms)

        # self.packet_handler = SMS_STS_PacketHandler(self.port_handler)
        self.packet_handler = ProtocolPacketHandler(port_handler=self.port_handler, byte_order=ByteOrder.LITTLE)

        self._sync_readers: Dict[Tuple[int, int], GroupSyncReader] = {}
        self._sync_writers: Dict[Tuple[int, int], GroupSyncWriter] = {}

        # NOTE: data array is indexed by idx of self._motor_ids, not by the id value.
        cached_value_names = [ TableValueName.pos, TableValueName.vel, TableValueName.load, TableValueName.vin] # ['pos', 'vel', 'load', 'vin']
        # self._cached_read_data_dict: Dict[TableValueName, npt.NDArray[Any]] =  {_name : None for _name in cached_value_names}
        self._cached_read_data_dict: Dict[TableValueName, npt.NDArray[Any]] =  {
            _name : np.full(shape=len(self._motor_ids), fill_value=np.nan, dtype=_TableValueReadSpec[_name].result_dtype)
            for _name in cached_value_names
        }

        # self._cur_scale_arr: Optional[npt.NDArray[np.float32]] = None

        # func wrappers:
        # self.sync_write_1_byte = partial(self._sync_write_impl, size=1)
        # self.sync_write_2_bytes = partial(self._sync_write_impl, size=2)
        # self.sync_write_3_bytes = partial(self._sync_write_impl, size=3)
        # self.sync_write_4_bytes = partial(self._sync_write_impl, size=4)

        # self.sync_read_1_byte = partial(self._sync_read_impl, size=1)
        # self.sync_read_2_bytes = partial(self._sync_read_impl, size=2)
        # self.sync_read_4_bytes = partial(self._sync_read_impl, size=4)
        # self.sync_read_6_bytes = partial(self._sync_read_impl, size=6)

        FeiteGroupClient.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self):
        """Connects to the Feite motors.

        NOTE: This should be called after all FeiteClients on the same
            process are created.
        """
        assert not self.is_connected, "Client is already connected."

        if self.port_handler.openPort():
            logger.info(f"Succeeded to open port: {self.port_name}")
        else:
            raise OSError(
                (
                    "Failed to open port at {} (Check that the device is powered "
                    "on and connected to your computer)."
                ).format(self.port_name)
            )

        if self.port_handler.setBaudRate(self.baud_rate):
            logger.info(f"Succeeded to set baud_rate to {self.baud_rate}")
        else:
            raise OSError(
                (
                    "Failed to set the baudrate to {} (Ensure that the device was "
                    "configured for this baudrate)."
                ).format(self.baudrate)
            )

        # Start with all motors enabled.  NO, I want to set settings before enabled
        # self.set_torque_enabled(self._motor_ids, True)

    def disconnect(self):
        """Disconnects from the Feite device."""
        if not self.is_connected:
            return
        if self.port_handler.is_using:
            logger.error("Port handler in use; cannot disconnect.")
            return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(self._motor_ids, False)
        self.port_handler.closePort()
        if self in FeiteGroupClient.OPEN_CLIENTS:
            FeiteGroupClient.OPEN_CLIENTS.remove(self)



    def _sync_read_impl(
        self, *, address:int, size: int,
            #, scale: float
    ) -> Tuple[float,List[bytearray|None]]:
        """Reads values from a group of motors.

        Args:
            _motor_ids: The motor IDs to read from.
            address: The control table address to read from.
            size: The size of the control table value being read.

        Returns:
            The values read from the motors.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_readers:
            self._sync_readers[key] = GroupSyncReader(
                packet_handler=self.packet_handler,
                start_address=address,
                data_length=size)

            # only add param once, causing we will not change _motor_ids after instantiate of `self`.
            for _id in self._motor_ids:
                # we never clear sync reader's param.
                if not self._sync_readers[key].addParam(_id):
                    raise OSError(
                        "[Motor ID: {}] Could not add parameter to sync read.".format(_id)
                    )

        sync_reader: GroupSyncReader = self._sync_readers[key]

        comm_time = 0.0
        success = False
        # TODO: count failure and return.
        while not success:
            # fastSyncRead does not work for 2XL and 2XC
            # time_1 = time.time()
            comm_time = time.time()
            comm_result = sync_reader.txRxPacket()

            # time_2 = time.time()
            # print(f"RTT: {time_2 - time_1}")

            success = self.handle_packet_result(comm_result.value, context="sync_read")
            #TODO: if not success, sleep 10ms.
            time.sleep(0.01)

        errored_ids: List[int] = []
        # data_dict : Dict[int, bytearray] = {}  # [[] for _ in range(len(self._motor_ids))]     # = np.zeros(len(self._motor_ids), dtype=np.float32)
        # NOTE: data array is indexed by idx of self._motor_ids, not by the id value.
        param_list : List[bytearray|None] = [None for _ in range(len(self._motor_ids)) ]

        for _x, _id in enumerate(self._motor_ids):
            # Check if the data is available.
            if not sync_reader.isAvailable(_id, address, size):
                # corresponding data of error id in param_list keeps as `None`.
                errored_ids.append(_id)
                continue

            # data_dict[_id]=sync_reader.getDataAsBytes(_id, address, size)
            param_list[_x]=sync_reader.getDataAsBytes(scs_id=_id, address=address, size=size)

        if errored_ids:
            # TODO: add handel for failure read: we can not get pos/vel states from actuators, which is
            # dangerous for controlling. maybe we should halt any action?
            logger.error( f"Sync read failed for: {str(errored_ids)}")
            raise ValueError(f"Sync read failed for: {str(errored_ids)}")

        # TODO: cause we will not change _motor_ids after instantiate of `self`,  so no need to call clearParam.
        # recv_data_dict,param are set/cleared after every sync_read.
        # sync_reader.clearParam()

        return comm_time, param_list


    def check_connected(self):
        """Ensures the robot is connected."""
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError("Must call connect() first.")

    def handle_packet_result(
        self,
        comm_result: int,
        motor_error: Optional[int] = None,
        motor_id: Optional[int] = None,
        context: Optional[str] = None,
    ) -> bool:
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != CommResult.SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif motor_error is not None:
            error_message = self.packet_handler.getRxPacketError(motor_error)

        if error_message:
            if motor_id is not None:
                error_message = "[Motor ID: {}] {}".format(motor_id, error_message)
            if context is not None:
                error_message = "> {}: {}".format(context, error_message)

            logger.error(error_message)

            return False

        return True

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
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self._write_1_byte_impl(motor_ids=remaining_ids,
                                                    value=int(enabled),
                                                    address=SMS_STS_SRAM_Table_RW.TORQUE_ENABLE)

            if remaining_ids:
                logger.error(f"Could not set torque {'enabled' if enabled else 'disabled'} for IDs: {str(remaining_ids)}")
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def _sync_read_helper(self,read_spec:_ReadSpec )\
            ->Tuple[float, List[npt.NDArray[Any]] ]:
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
                assert len(_bytes) == sum(read_spec.size)
                for _s, _psr, _arr in zip(read_spec.size, read_spec.parser, value_arr_list):
                    # bytearray(_bytes.pop(0) for _ in range(_s))
                    _arr[_x] = _psr(_bytes[0:_s])
                    del _bytes[0:_s]

        # NOTE: if the ID does not return pkt, the corresponding value is `0`.
        return comm_time, value_arr_list

    # def read_model_number(self, retries: int = 0) -> npt.NDArray[np.uint16]:
    #     """Returns the model number of the motors."""
    #     param_list: List[bytearray|None]
    #     _, param_list = self.sync_read_2_bytes(address=SMS_STS_EEPROM_Table_ReadOnly.MODEL_L)
    #     assert len(param_list) == len(self._motor_ids)
    #
    #     model_numbers = np.zeros(shape= len(self._motor_ids), dtype=np.uint16)
    #
    #     # TODO: corresponding data of error id in param_list keeps as `None`. check the valid value ?
    #     for _x, _bytes in enumerate(param_list):
    #         value = None
    #         if _bytes is None:
    #             value = 0
    #             raise ValueError(f'the read data of ID:{self._motor_ids[_x]} is None.')
    #         else:
    #             assert len(_bytes) == 2
    #             value = int.from_bytes(_bytes, byteorder='little',signed=False)
    #
    #         model_numbers[_x] = value
    #
    #     # NOTE: if the ID does not return version, the corresponding version value is `0`.
    #     return model_numbers

    def _read_table_single_value_helper(self,*, name: TableValueName, into_cache: bool, retries: int = 0) -> Tuple[float,npt.NDArray[Any]]:
        comm_time, value_arr_list = self._sync_read_helper(_TableValueReadSpec[name])
        assert len(value_arr_list)==1 and len(value_arr_list)==len(_TableValueReadSpec[name].result_dtype)
        if into_cache:
            assert self._cached_read_data_dict[name].dtype == _TableValueReadSpec[name].result_dtype
            self._cached_read_data_dict[name] = value_arr_list[0].copy()

        # NOTE: if the ID does not return pkt, the corresponding value is `np.nan`.
        return comm_time, value_arr_list[0]

    def _read_table_multiple_values_helper(self,*, name: TableValueName, retries: int = 0) -> Tuple[float,List[npt.NDArray[Any]] ]:
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

    # def read_pos(self, retries: int = 0) -> Tuple[float,npt.NDArray[np.float32]]:
    #     """Returns the current positions."""
    #     value_name: str = 'pos'
    #     comm_time, self._cached_read_data_dict[value_name] = self.sync_read_helper(_TableValueReadSpec[value_name])
    #
    #     # NOTE: if the ID does not return pkt, the corresponding value is `np.inf`.
    #     return comm_time, self._cached_read_data_dict[value_name].copy()

    def read_vel(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        return self._read_table_single_value_helper(name=TableValueName.vel, into_cache=True)

    # def read_vel(self, retries: int = 0) -> Tuple[float,npt.NDArray[np.float32]]:
    #     """Returns the current velocities."""
    #     attr: str = 'vel'
    #     comm_time, self._cached_read_data_dict[attr] = self.sync_read_helper(addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VELOCITY_L,
    #                                                                           size=2,
    #                                                                           parser=_parse_vel)
    #
    #     # NOTE: if the ID does not return pkt, the corresponding value is `np.inf`.
    #     return comm_time, self._cached_read_data_dict[attr].copy()

    def read_vin(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float16]]:
        return self._read_table_single_value_helper(name=TableValueName.vin, into_cache=True)

    #
    # def read_vin(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
    #     """Reads the input voltage to the motors."""
    #     attr:str = 'vin'
    #     comm_time, self._cached_read_data_dict[attr] = self.sync_read_helper(addr=SMS_STS_SRAM_Table_ReadOnly.PRESENT_VOLTAGE,
    #                                                                           size=1,
    #                                                                           parser=_parse_vin)
    #
    #     # NOTE: if the ID does not return pkt, the corresponding value is `np.inf`.
    #     return comm_time, self._cached_read_data_dict[attr].copy()

    def read_pos_vel_load(
        self, retries: int = 0
    ) -> PosVelLoadRecord:
        comm_time, value_arr_list = self._read_table_multiple_values_helper(name=TableValueName.pos_vel_load)

        # TODO: the necessary of saving into cached data dict?
        self._cached_read_data_dict[TableValueName.pos]=value_arr_list[0]
        self._cached_read_data_dict[TableValueName.vel]=value_arr_list[1]
        self._cached_read_data_dict[TableValueName.load]=value_arr_list[2]  # load in percentage of stall torque.

        return PosVelLoadRecord(
            comm_time,
            pos=self._cached_read_data_dict[TableValueName.pos].copy(),
            vel=self._cached_read_data_dict[TableValueName.vel].copy(),
            load=self._cached_read_data_dict[TableValueName.load].copy(),
        )

    # def read_pos_vel_load(
    #     self, retries: int = 0
    # ) -> PosVelLoadRecord:
    #     # NEED to update line 115 and 349 if calling this function
    #     """Returns the current positions and velocities, and torque"""
    #     param_list: List[bytearray]
    #
    #     # read addr 56,58,60.
    #     comm_time, param_list = self.sync_read_6_bytes(address=SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L)
    #                                                   #size= SMS_STS_SRAM_Table_ReadOnly.PRESENT_LOAD_H - SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L + 1) #  6, )
    #
    #     assert len(param_list) == len(self._motor_ids)
    #
    #     # TODO: corresponding data of error id in param_list keeps as `None`. check the valid value ?
    #     for _x, _bytes in enumerate(param_list):
    #         assert len(_bytes) == 6
    #         # get pos.
    #         self._cached_read_data_dict["pos"][_x] = _parse_pos(_bytes[0:2])
    #         # get vel
    #         self._cached_read_data_dict["vel"][_x] = _parse_vel(_bytes[2:4])
    #         # get load, percentage of stall torque.
    #         self._cached_read_data_dict["load"][_x] = _parse_load(_bytes[4:6])
    #
    #     # NOTE: if the ID does not return pkt, the corresponding value is timed value...
    #     return PosVelLoadRecord(
    #         comm_time,
    #         pos=self._cached_read_data_dict["pos"].copy(),
    #         vel=self._cached_read_data_dict["vel"].copy(),
    #         load=self._cached_read_data_dict["load"].copy(),
    #     )


    def _sync_write_impl(
        self,*,
        param: List[bytes|bytearray] | bytes| bytearray,
        address: int,
        # size: int,
        write_ids: Optional[Sequence[int]] = None,
    ):
        """Writes values to a group of motors.

        Args:
            write_ids: The motor IDs to write to.
            param: The values to write. single bytes/bytearray value means same value for all ID, a list(bytes|bytearray) contains
                   individual value for every ID.
            address: The control table address to write to.
            #size: The size(in bytes) of the control table value being written to. can cover multiple registers.
        """
        size: int
        errored_ids: List[int] = []

        if isinstance(param,list):
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

        if isinstance(param,list):
            assert len(write_ids) == len(param)

        # Clear before addParam.
        sync_writer.clearParam()

        if isinstance(param,list):
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
        self.handle_packet_result(comm_result.value, context="sync_write")

        # write_data_dict,param are set/cleared at every sync_write.
        # sync_writer.clearParam()


    def _write_and_recv_answer_impl(
        self,*,
        param_list: List[bytes|bytearray], # each element is for corresponding id.
        address: int,    # same for all ids.
        write_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Writes a value to the motors.
           vs. sync_write:  we need the individual feedback of corresponding ID here, but sync_write has no feedback pkt.

        Args:
            write_ids: The motor IDs to write to.
            param_list: The value to write to the control table.
            address: The control table address to write to. same for all id.

        Returns:
            A list of IDs that were unsuccessful.
        """
        errored_ids: List[int] = []
        size: int = len(param_list[0])  # should be same for all elements in params.

        self.check_connected()

        if write_ids is None:
            write_ids = self._motor_ids

        assert len(write_ids) == len(param_list)

        for _id, _bytes in zip(write_ids, param_list):
            assert len(_bytes) == size  # should be same for all elements in params.
            comm_result,error = self.packet_handler.writeTxRx(
                scs_id = _id,
                address = address,
                length = len(_bytes) ,
                data = _bytes)
            success = self.handle_packet_result(
                comm_result.value,
                error,
                _id,
                context="_write_and_recv_answer_impl",
            )
            if not success:
                errored_ids.append(_id)
        return errored_ids


    def _set_1_byte_param_helper(self, *, address:int, value:int|List[int]):
        param: bytes | List[bytes]

        if isinstance(value, list):
            for _v in value:
                assert 0 <= _v <= 0xff

            param = [_v.to_bytes(length=1) for _v in value]
        else:
            assert 0 <= value <= 0xff
            param = value.to_bytes(length=1)

        self._sync_write_impl(param=param,
                              address=address)


    def _set_multiple_bytes_param_helper(self,*, address:int, value: Tuple[int,...] | Sequence[Tuple[int,...]]):
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

    def set_desired_pos( self, *, motor_ids: Sequence[int], positions: npt.NDArray[np.float32] ):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write. in rad of single turn.signed value, to represent rotor direction.
        """
        assert len(motor_ids) == len(positions)
        # TODO: only allow -2Pi ~ 2Pi.
        assert np.all( np.abs(positions) < 2*np.pi )

        # ->steps, ->int, signed->unsigned, to_bytes.
        # Convert to Feite position steps:
        steps = (positions / POS_RESOLUTION).astype(dtype=np.int16)
        size = steps.dtype.itemsize
        assert size==2

        self._sync_write_impl(param=[_signed_to_proto_param_bytes_v2(value=_v, size=size)
                                          for _v in steps],  # addr 42, 43,
                              address=SMS_STS_SRAM_Table_RW.GOAL_POSITION_L,
                              write_ids=motor_ids)

        # self.sync_write_2_bytes(_motor_ids=_motor_ids,
        #                         params=[_signed_to_proto_param_bytes_v2(value=_v, size=2) for _v in steps],  # addr 42, 43,
        #                         address=SMS_STS_SRAM_Table_RW.GOAL_POSITION_L)  # addr 42, 43


    def set_return_delay_time(self, value: int|List[int]):
        # TODO: value >= 1.
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.RETURN_DELAY_TIME, value=value)

    def set_control_mode(self, value: int|List[int]):
        # TODO: only allow 0,1,2,3
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.CONTROL_MODE,value=value)

    def set_kp(self, value: int | List[int]):
        #TODO: not allow 0 value for kp....
        self._set_1_byte_param_helper(address=SMS_STS_EEPROM_Table_RW.KP, value=value)

    # def set_kp_kd(self, *, kp: int | List[int], kd: int|List[int]):
    #     assert 0 < kp < 0xff
    #     assert 0 < kd < 0xff
    #
    #     # kp:21, kd:22 consecutive address.
    #     self._sync_write_impl(param=bytes([kp, kd]),
    #                           address=SMS_STS_EEPROM_Table_RW.KP)

    def set_kp_kd_ki(self, *, kp: List[int], kd: List[int], ki: List[int]):
        # kp:21, kd:22 , ki:23 consecutive address.
        self._set_multiple_bytes_param_helper(address=SMS_STS_EEPROM_Table_RW.KP,
                                              value=list(zip(kp, kd, ki)) )

    # TODO: SMS/STS has no reboot function
    # def reboot(self, _motor_ids: Sequence[int]):
    #     """Reboots the specified motors.
    #
    #     Args:
    #         _motor_ids (Sequence[int]): A sequence of motor IDs to reboot.
    #     """
    #     for motor_id in _motor_ids:
    #         self.packet_handler.reboot(self.port_handler, motor_id)

    # def convert_to_unsigned(self, value: int, size: int) -> int:
    #     """Converts the given value to its unsigned representation."""
    #     if value < 0:
    #         max_value = (1 << (8 * size)) - 1
    #         value = max_value + value
    #     return value

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


# Register global cleanup function.
atexit.register(_feite_cleanup_handler)
