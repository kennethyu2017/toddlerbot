from numpy import pi
from copy import deepcopy
from .protocol_packet_handler import ProtocolPacketHandler
from .sms_sts import  (SMS_STS_SRAM_Table_ReadOnly,
                       ACCEL_RESOLUTION, POS_RESOLUTION,VEL_RESOLUTION,
                       LOAD_PERCENTAGE_RESOLUTION, VOLTAGE_RESOLUTION,
                       SMS_STS_SRAM_Table_RW, CommResult)


# TODO: for feite protocol, the Two’s complement is not applied for the negative value. Use the BIT15 instead...
def signed_to_proto_param_bytes_v2(*, value: int, size: int) -> bytes | bytearray:
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

def proto_param_bytes_to_signed_v2(*, param: bytes | bytearray) -> int:
    """Converts an unsigned integer to a signed integer of a specified byte size.

    Args:
        param (bytes): The unsigned integer to convert. the highest bit, e.g., BIT15, representing sign.

    Returns:
        int: The signed integer representation of the input value.
    """

    # use the highest bit, e.g., BIT15, to represent sign.
    # TODO: check the real act of actuator.
    sign:int = 1

    if (param[-1] & 0x80) !=0 :   # (1 << 7) got unsigned int.
        sign = -1
        # NOTE: deepcopy to avoid modifying argument `param` buffer.
        param = deepcopy(param)

        if isinstance(param,bytes):
            param = bytearray(param)

        param[-1] &= 0x7f  # ~(1 << 7) got unsigned int.

    return sign * int.from_bytes(bytes=param, byteorder='little', signed=False)


def parse_model(param: bytes | bytearray) -> int:
    assert len(param) == 2
    return int.from_bytes(bytes=param, byteorder='little', signed=False)

def parse_pos(param: bytes | bytearray)->float:
    assert len(param) == 2
    # feedback param is present absolute steps in a single turn, w/o direction.
    step = int.from_bytes(bytes=param, byteorder='little', signed=False)
    assert 0 <= step <= 4095
    return step * POS_RESOLUTION  # rad in as single turn.

def parse_vel(param: bytes | bytearray)->float:
    assert len(param) == 2
    # BIT15 is direction.
    vel = proto_param_bytes_to_signed_v2(param=param)
    # vel resolution is 0.732 rpm, SM40BL max vel is 88 rpm.
    assert -120 <= vel <= 120
    return vel * VEL_RESOLUTION

def parse_load(param: bytes | bytearray)->float:
    assert len(param) == 2
    # 0.1% of stall_torque.
    load = int.from_bytes(bytes=param, byteorder='little', signed=False)
    # assert 0 <= load <= 1000
    if not 0 <= load <= 1000:
        pass
        # raise ValueError(f'read motor load value error: {load}')
        # print(f' WARING: ====  read motor load value error: {load}')
        # print(f' WARING: ====  read motor load value error: {load}')
        # print(f' WARING: ====  read motor load value error: {load}')


    return load * LOAD_PERCENTAGE_RESOLUTION


def parse_vin(param: bytes | bytearray)->float:
    assert len(param) == 1
    vin = int.from_bytes(bytes=param, byteorder='little', signed=False)
    assert 0 <= vin < 140
    return vin * VOLTAGE_RESOLUTION


def read_pos_and_vel_helper(reader: ProtocolPacketHandler, motor_id: int):
    # Read the present pos and vel.
    pos_and_vel, comm_result, error = reader.readTxRx(motor_id,
                                                      SMS_STS_SRAM_Table_ReadOnly.PRESENT_POSITION_L,
                                                      4)
    assert len(pos_and_vel) == 4
    if comm_result != CommResult.SUCCESS:
        raise IOError(f'reader.readTxRx comm error: {reader.getTxRxResult(comm_result)} ')
    else:
        print(f'Read [ID: {motor_id:>3d}] --->'
              f' Present Pos:{parse_pos(pos_and_vel[:2]): 4.4f}'
              f' Present Vel:{parse_vel(pos_and_vel[2:]): 4.4f}')

    if error != 0:
        raise ValueError(f'got motor error: {reader.getRxPacketError(error)} ')

def write_acc_pos_vel_helper(*, writer: ProtocolPacketHandler,
                              motor_id:int,
                              acc_radius:float,
                              pos_radius:float,
                              vel_radius:float):
    # comm_result, error = packet_handler.WritePosEx(1, 4095, 60, 50)
    txpkt: bytes = construct_acc_pos_vel_txpkt_helper(acc_radius=acc_radius,
                                                      pos_radius=pos_radius,
                                                      vel_radius=vel_radius)

    comm_result, error = writer.writeTxRx(scs_id=motor_id,
                                          address=SMS_STS_SRAM_Table_RW.ACC,
                                          length=len(txpkt),
                                          data=txpkt)

    if comm_result != CommResult.SUCCESS:
        raise IOError(f'writer.writeTxRx comm error: {writer.getTxRxResult(comm_result)}')

    if error != 0:
        raise ValueError(f'got motor error: {writer.getRxPacketError(error)} ')


def construct_acc_pos_vel_txpkt_helper(*, acc_radius: float,
                                       pos_radius: float,
                                       vel_radius:float)->bytes:

    # TODO: only allow -2Pi ~ 2Pi.
    assert abs(pos_radius) < (2 * pi)
    # Convert to Feite position steps:
    pos_steps: int = round(pos_radius / POS_RESOLUTION)
    assert abs(pos_steps) < 4096

    # vel resolution is 50 steps / second, i.e.,  0.0767 rad/s,  0.732 rpm/s., SM40BL max vel is 88 rpm.
    assert abs(vel_radius) < 9
    vel_50_steps: int = round(vel_radius / VEL_RESOLUTION)
    assert abs(vel_50_steps) < 117

    # acc resolution is  0.1534 rad/s2,  1.465 rpm/s2., SM40BL max vel is 88 rpm.
    assert abs(acc_radius) < 7.6
    acc_100_steps: int = round(acc_radius / ACCEL_RESOLUTION)
    assert abs(acc_100_steps) < 50

    print(f'{pos_steps=:}  {vel_50_steps=:}  {acc_100_steps=:} ')

    txpacket = bytes(
        [*signed_to_proto_param_bytes_v2(value=acc_100_steps, size=1),  # result is little-endian.
         *signed_to_proto_param_bytes_v2(value=pos_steps, size=2),
         *signed_to_proto_param_bytes_v2(value=0, size=2),  # run time. ignore.
         *signed_to_proto_param_bytes_v2(value=vel_50_steps, size=2)]
    )

    assert len(txpacket) == 7

    return txpacket
