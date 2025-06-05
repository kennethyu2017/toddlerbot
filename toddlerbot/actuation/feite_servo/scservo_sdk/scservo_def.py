#!/usr/bin/env python

from enum import Enum

BROADCAST_ID = 0xFE  # 254
MAX_ID = 0xFC  # 252
SCS_END = 0

# Instruction for SCS Protocol
# @dataclass(init=True)
class SCSProtoInst:
    PING : int = 1
    READ: int = 2
    WRITE = 3
    REG_WRITE = 4
    ACTION = 5
    SYNC_WRITE = 131  # 0x83
    SYNC_READ = 130  # 0x82

# Communication Result. using Enum type for easy repr.
class CommResult(Enum):
    SUCCESS : int = 0  # tx or rx packet communication success
    PORT_BUSY : int = -1  # Port is busy (in use)
    TX_FAIL : int = -2  # Failed transmit instruction packet
    RX_FAIL = -3  # Failed get status packet
    TX_ERROR = -4  # Incorrect instruction packet
    RX_WAITING = -5  # Now recieving status packet
    RX_TIMEOUT = -6  # There is no status packet
    RX_CORRUPT = -7  # Incorrect status packet
    NOT_AVAILABLE = -9  #

