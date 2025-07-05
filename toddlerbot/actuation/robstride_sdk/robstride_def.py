from typing import Dict,Any,Type, NamedTuple, Tuple
from numpy import pi

ROBSTRIDE_DEFAULT_BAUD_RATE = 1_000_000

class _ReadSpec(NamedTuple):
    index: int
    n_bytes: int
    # parser: Sequence[ Callable[[bytes|bytearray], float|int] ]
    dtype: Type
    min_max: Tuple[float, float] | None = None


# TODO: this is only for RS02. check other types.
_ParamTableReadSpec : Dict[str, _ReadSpec] = {
    'run_mode': _ReadSpec(index=0x7005,
                          n_bytes=1,
                          dtype=int),

    'limit_torque': _ReadSpec(index=0x700B,
                               n_bytes=4,
                               dtype=float,
                               min_max=(0., 17.) ),

    # target pos in PP mode./CSP mode.
    'loc_ref': _ReadSpec(index=0x7016,
                         n_bytes=4,
                         dtype=float,
                         min_max=(-12.57, 12.57) ),

    'mechPos': _ReadSpec(index=0x7019,
                         n_bytes=4,
                         dtype=float,
                         min_max=(-12.57, 12.57)),

    'mechVel': _ReadSpec(index=0x701B,
                         n_bytes=4,
                         dtype=float,
                         min_max=(-44., 44.)),

    'loc_kp': _ReadSpec(index=0x701E,
                        n_bytes=4,
                        dtype=float,
                        min_max=(0, 200)),

    'spd_kp': _ReadSpec(index=0x701F,
                        n_bytes=4,
                        dtype=float,
                        min_max=(0,200)),

    # vel max abs value in PP mode.
    'vel_max': _ReadSpec(index=0x7024,
                         n_bytes=4,
                         dtype=float,
                         min_max=(0, 44.)),

    # acc abs value in PP mode.
    'acc_set': _ReadSpec(index=0x7025,
                         n_bytes=4,
                         dtype=float,
                         # TODO> max acc of RS?
                         min_max=(0, 30.)),

    'EPScan_time': _ReadSpec(index=0x7026,
                             n_bytes=2,
                             dtype=int,
                             min_max=(0., 50.)),

    'zero_sta': _ReadSpec(index=0x7029,
                          n_bytes=1,
                          dtype=int),
}

# for RobStride private protocol:
class CommunicationType:
    GET_DEVICE_ID = 0
    MOTION_CONTROL = 1
    MOTOR_FEEDBACK = 2
    MOTOR_ENABLE = 3
    MOTOR_STOP = 4
    SET_MECH_POS_ZERO = 6
    SET_MOTOR_CAN_ID = 7
    # PARAM_TABLE_WRITE: int = 8
    SINGLE_PARAM_READ = 17
    SINGLE_PARAM_WRITE = 18  # will get lost after power down if not use type 22.
    ERROR_FEEDBACK = 21
    SAVE_PARAM = 22
    SET_BAUD_RATE = 23
    MOTOR_PERIODIC_REPORT = 24
    SET_PROTOCOL = 25

# index: 0x7005
class RunModes:
    MOTION_MODE = 0        # 运控模式
    PP_POSITION_MODE = 1   # PP位置模式
    SPEED_MODE = 2         # 速度模式
    CURRENT_MODE = 3       # 电流模式
    CSP_POSITION_MODE = 5  # CSP位置模式

# TODO: this is for RS02 only.
class ParamThreshold:
    P_MIN = -12.57
    P_MAX = 12.57
    V_MIN = -44.0
    V_MAX = 44.0
    T_MIN = -17.0
    T_MAX = 17.0
    KP_MIN, KP_MAX = (0.0, 500.0)
    KD_MIN, KD_MAX = (0.0, 5.0)

# comm type: 23
class BaudRate:
    BPS_1M = 1
    BPS_500K = 2
    BPS_250K = 3
    BPS_125K = 4
