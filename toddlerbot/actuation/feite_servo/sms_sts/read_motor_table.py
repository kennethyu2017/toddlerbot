#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS), and an URT
#
import time
from toddlerbot.actuation.feite_servo.scservo_sdk import (PortHandler,SMS_STS_DEFAULT_BAUD_RATE,
                                                          ProtocolPacketHandler, ByteOrder,
                                                          proto_param_bytes_to_signed_v2,
                                                          CommResult, )

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'

def read_addr_helper(*, pkt_handler: ProtocolPacketHandler,
                     motor_id: int, addr: int, length: int):
    # Read the present pos and vel.
    rxpkt, comm_result, error = pkt_handler.readTxRx(scs_id=motor_id, address=addr, length=length)

    assert len(rxpkt) == length

    if comm_result != CommResult.SUCCESS:
        raise IOError(f'pkt_handler.readTxRx return is not success ,error info:  {pkt_handler.getTxRxResult(comm_result)} ')

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
        raise  ValueError(f'pkt_handler.readTxRx got error from rcv pkt: {pkt_handler.getRxPacketError(error)} ')


def _input_int_value_helper(value_range: range, prompt: str) -> int:
    _value : int = value_range.stop # stop not in range.
    while not _value in value_range:
        try:
            _value = int(input(prompt))
        except ValueError as err:
            _value = value_range.stop
            print(f'key-in value error: {err}, {type(err)}')
            continue
        else:
            if _value not in value_range:
                print(f'got illegal value: {_value}, should be in scope: {value_range} ')
                _value = value_range.stop

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

    motor_id: int = -1
    addr: int = -1
    length: int = -1

    try:
        while True:
                time.sleep(1.)
                print(f'\n--- start read motor control table:')
                motor_id = _input_int_value_helper(range(0,254), f'\ninput read motor id [0~253] : ')
                addr = _input_int_value_helper(range(0, 70), f'\ninput read start addr [0~69] : ')
                length = _input_int_value_helper(range(1,11), f'\ninput read length of bytes [1~10]: ')

                read_addr_helper(pkt_handler=_packet_handler,motor_id=motor_id, addr=addr, length=length)

    finally:
        # Close port
        _port_handler.closePort()

if __name__  == '__main__':
    _main()