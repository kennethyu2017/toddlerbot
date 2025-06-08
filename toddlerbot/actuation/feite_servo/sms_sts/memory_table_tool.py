#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#
import warnings
from typing import Set
import time
from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          proto_param_bytes_to_signed_v2,
                                                          signed_to_proto_param_bytes_v2,
                                                          CommResult, )

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_ID_SET: Set[int] = {0}

def _write_addr_helper(*, writer: ProtocolPacketHandler, value: int,
                      motor_id: int, addr: int, length: int):

    # only support 1 ~ 2 bytes write.
    assert length in {1,2}
    if length == 1:
        assert -0xff < value <= 0xff
    elif length == 2:
        assert -0xffff < value <= 0xffff

    txpkt = signed_to_proto_param_bytes_v2(value=value, size=length)  # result is little-endian.
    assert len(txpkt) == length


    print(f'--- Write [ID: {motor_id:>2d}] addr:[ {addr:>2d}] length:[{length:>2d}]  txpkt --->')
    for _b in txpkt:
        print(f'0x{_b:02x}')

    print(f'--- end of txpkt.')

    comm_result, error = writer.writeTxRx(scs_id=motor_id,
                                          address=addr,
                                          length=len(txpkt),
                                          data=txpkt)

    if comm_result != CommResult.SUCCESS:
        # raise IOError(f'writer.writeTxRx comm error : {writer.getTxRxResult(comm_result)}')
        print(f'Warning: writer.writeTxRx comm error : {writer.getTxRxResult(comm_result)}, pls check the motor ID / motor connection.')

    if error != 0:
        raise ValueError(f'writer.writeTxRx got error from motor : {writer.getRxPacketError(error)} ')


def _read_addr_helper(*, reader: ProtocolPacketHandler,
                      motor_id: int, addr: int, length: int):
    # Read the present pos and vel.
    rxpkt, comm_result, error = reader.readTxRx(scs_id=motor_id, address=addr, length=length)

    if rxpkt is None:
        print(f'Warning: rxpkt is None, pls check the motor ID / motor connection.')
        return

    assert len(rxpkt) == length

    if comm_result != CommResult.SUCCESS:
        raise IOError(f'reader.readTxRx comm error:  {reader.getTxRxResult(comm_result)} ')

    else:
        print(f'--- Read [ID: {motor_id:>2d}] addr:[ {addr:>2d}] length:[{length:>2d}] result rxpkt --->')
        for _b in rxpkt:
            print(f'0x{_b:02x}')
        print(f'--- end of rxpkt.')

        if len(rxpkt) <= 2 :
            # highest bit represent sign.
            signed_dec_value:int = proto_param_bytes_to_signed_v2(param=rxpkt)

            unsigned_dec_value:int = int.from_bytes(rxpkt, byteorder='little',signed=False)
            print(f'+++ when length <=2 , we can parse rxpkt to signed decimal value: {signed_dec_value}, '
                  f'unsigned decimal value:{unsigned_dec_value}')

    if error != 0:
        raise ValueError(f'reader.readTxRx got error from motor: {reader.getRxPacketError(error)} ')


def _input_str_value_helper(valid_input: Set[str], prompt: str) -> str:
    r_w: str = ''

    while not r_w in valid_input:
        try:
            r_w = input(prompt)

        except ValueError as err:
            r_w = ''
            print(f'key-in value error: {err}, {type(err)}')
            continue

        else:
            if r_w not in valid_input:
                print(f'got illegal input: {r_w}, should be "r" or "w":  ')
                r_w =''

    return r_w


def _input_int_value_helper(*, legal_set: range | Set[int], illegal_sentinel: int, prompt: str) -> int:
    # _value : int = value_range.stop # stop not in range.
    _value = illegal_sentinel

    while not _value in legal_set:
        try:
            _value = int(input(prompt))

        except ValueError as err:
            # _value = value_range.stop
            _value = illegal_sentinel
            print(f'key-in value error: {err}, {type(err)}')
            continue

        else:
            if _value not in legal_set:
                print(f'got illegal input: {_value}, should be in set: {legal_set} ')
                # _value = value_range.stop
                _value = illegal_sentinel

    return _value


def _main():

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    _port_handler = PortHandler(port_name=URT_1_DEV_NAME,
                               baud_rate=SMS_STS_DEFAULT_BAUD_RATE,
                               rcv_timeout_ms=10)  # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    # Initialize PacketHandler instance
    # Get methods and members of Protocol
    _packet_handler = ProtocolPacketHandler(port_handler=_port_handler,
                                           byte_order=ByteOrder.LITTLE)

    # Open port
    if _port_handler.openPort():
        print("Succeeded to open the port")
    else:
        raise IOError("Failed to open the port")

    read_or_write : str = ''
    motor_id: int = -1
    addr: int = -1
    length: int = -1

    try:
        while True:

                time.sleep(1.)

                read_or_write = _input_str_value_helper( {'r', 'w'}, f'\nread or write? [r/w] : ')

                if read_or_write == 'r':
                    print(f'\n--- start READ motor control table:')
                    motor_id = _input_int_value_helper( legal_set=MOTOR_ID_SET,illegal_sentinel=-1,
                                                        prompt=f'\ninput read motor id in choices {MOTOR_ID_SET} : ')

                    addr = _input_int_value_helper(legal_set=range(0, 70), illegal_sentinel=-1,
                                                   prompt='\ninput read start addr [0~69] : ')

                    length = _input_int_value_helper(legal_set=range(1,11), illegal_sentinel=-1,
                                                     prompt=f'\ninput read length of bytes [1~10]: ')

                    _read_addr_helper(reader=_packet_handler, motor_id=motor_id, addr=addr, length=length)

                elif read_or_write == 'w':
                    print(f'\n--- start WRITE motor control table:')
                    motor_id = _input_int_value_helper( legal_set=MOTOR_ID_SET, illegal_sentinel=-1,
                                                        prompt=f'\ninput write motor id in choices {MOTOR_ID_SET} : ')

                    addr = _input_int_value_helper(legal_set=range(0, 70), illegal_sentinel=-1 ,
                                                   prompt=f'\ninput write start addr [0~69] : ')

                    length = _input_int_value_helper(legal_set=range(1, 3),illegal_sentinel=-1,
                                                     prompt=f'\ninput write length of bytes [1~2], not support more than 2 bytes write : ')

                    value_upper = ( (1 << 8 * length) - 1 ) // 2
                    value = _input_int_value_helper( legal_set=range( -value_upper, value_upper), illegal_sentinel= value_upper + 88,
                                                     prompt=f'\ninput write value in {range( -value_upper, value_upper)} : ')
                    _write_addr_helper(writer=_packet_handler, value=value, motor_id=motor_id, addr=addr,length=length )

                else:
                    # raise ValueError(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                    print(f'operation mode error, should only be "r" or "w", but got: {read_or_write}')
                    continue

    finally:
        # Close port
        _port_handler.closePort()

if __name__  == '__main__':
    _main()