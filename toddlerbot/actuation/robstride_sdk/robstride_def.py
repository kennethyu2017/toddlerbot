from typing import Dict,Type, NamedTuple, Tuple

# ROBSTRIDE_DEFAULT_BAUD_RATE = 1_000_000

param_table_index_to_name: Dict[int,str] = {
    0X7005:'run_mode',
    0X700B:'limit_torque',
    0X7016:'loc_ref',
    0X7019:'mechPos',
    0X701B:'mechVel',
    0X701C:'VBUS',       # voltage.
    0X701E:'loc_kp',
    0X701F:'spd_kp',
    0X7024:'vel_max',
    0X7025:'acc_set',
    0X7026:'EPScan_time',
    0X7029:'zero_sta',
}

class ExtID(NamedTuple):
    dest_can_id: int
    data2: int
    comm_type: int

class MotorStateFrame(NamedTuple):
    ts: float  # time stamp.
    can_id: int
    pos: float
    vel: float
    torque: float
    temp: float    #temp_celsius
    # todo: motor error...

    def __str__(self):
        return (f'motor state frame --> timestamp:{self.ts:.6f} can_id:{self.can_id} pos:{self.pos:.2f} vel:{self.vel:.2f} '
                f'torque:{self.torque:.2f} temp:{self.temp:.2f}')

class SingleParamValue(NamedTuple):
    ts: float # timestamp
    can_id: int
    index: int
    value: float|int

    def __str__(self):
        if isinstance(self.value,float):
            return f'single param value --> timestamp:{self.ts:.6f} can_id: {self.can_id} index: 0x{self.index:x} value: {self.value:.2f}'
        else:
            return f'single param value --> timestamp:{self.ts:.6f} can_id: {self.can_id} index: 0x{self.index:x} value: {self.value:d}'


class ParamSpec(NamedTuple):
    index: int
    n_bytes: int
    # parser: Sequence[ Callable[[bytes|bytearray], float|int] ]
    dtype: Type[int|float]
    signed: bool
    min_max: Tuple[float|int, float|int]


# TODO: this is only for RS02. check other types.
RS_param_table_spec : Dict[str, ParamSpec] = {
    'run_mode': ParamSpec(index=0x7005,
                          n_bytes=1,
                          dtype=int,
                          signed=False,
                          min_max=(0, 5)),

    'limit_torque': ParamSpec(index=0x700B,
                              n_bytes=4,
                              dtype=float,
                              signed=True,
                              min_max=(0., 17.)),

    # target pos in PP mode./CSP mode.
    'loc_ref': ParamSpec(index=0x7016,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         min_max=(-12.57, 12.57)),

    'mechPos': ParamSpec(index=0x7019,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         min_max=(-12.57, 12.57)),

    'mechVel': ParamSpec(index=0x701B,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         min_max=(-44., 44.)),

    # voltage. read only.
    'VBUS': ParamSpec(index=0x701C,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         min_max=(0., 50.)),

    'loc_kp': ParamSpec(index=0x701E,
                        n_bytes=4,
                        dtype=float,
                        signed=True,
                        min_max=(0., 200.)),

    'spd_kp': ParamSpec(index=0x701F,
                        n_bytes=4,
                        dtype=float,
                        signed=True,
                        min_max=(0., 200.)),

    # vel max abs value in PP mode.
    'vel_max': ParamSpec(index=0x7024,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         min_max=(0., 44.)),

    # acc abs value in PP mode.
    'acc_set': ParamSpec(index=0x7025,
                         n_bytes=4,
                         dtype=float,
                         signed=True,
                         # TODO> max acc of RS?
                         min_max=(0., 200.)),

    'EPScan_time': ParamSpec(index=0x7026,
                             n_bytes=2,
                             dtype=int,
                             signed=False,
                             min_max=(0, 50)),

    'zero_sta': ParamSpec(index=0x7029,
                          n_bytes=1,
                          dtype=int,
                          signed=False,
                          min_max=(0, 1)),

}

# for RobStride private protocol:
class CommunicationType:
    GET_DEVICE_ID = 0
    MOTION_CONTROL = 1
    MOTOR_FEEDBACK = 2
    MOTOR_ENABLE = 3
    MOTOR_DISABLE = 4
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

# comm type: 23
class BaudRateCmd:
    BPS_1M = 1
    BPS_500K = 2
    BPS_250K = 3
    BPS_125K = 4

# param table index: 0x7005
class RunModeCmd:
    MOTION = 0  # 运控模式
    PP_POSITION = 1  # PP位置模式
    SPEED = 2  # 速度模式
    CURRENT = 3  # 电流模式
    CSP_POSITION = 5  # CSP位置模式

# param table index: 0x7026
class ReportPeriodCmd:
    # NOTE: RS data manual error: actually cmd value starting from 0
    P_10MS = 0
    P_15MS = 1
    P_20MS = 2
    P_25MS = 3
    P_30MS = 4
    P_35MS = 5
    P_40MS = 6
    P_45MS = 7

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


# ParamTableNamedIndex: Dict[str, int] = {
#     'run_mode': 0x7005,
#     'limit_torque': 0x700B,
#     # target pos in PP mode./CSP mode.
#     'loc_ref': 0x7016,
#     'mechPos': 0x7019,
#     'mechVel': 0x701B,
#     'loc_kp': 0x701E,
#     'spd_kp': 0x701F,
#     # vel max abs value in PP mode.
#     'vel_max': 0x7024,
#     # acc abs value in PP mode.
#     'acc_set': 0x7025,
#     'EPScan_time': 0x7026,
#     'zero_sta': 0x7029,
# }

