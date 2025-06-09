from dataclasses import dataclass,field
from typing import List, Dict, NamedTuple
import numpy as np
import numpy.typing as npt
from ..sim import Obs

RUN_POLICY_LOG_FOLDER_FMT = 'run_policy_log/{robot_name}_{policy_name}_{env_name}_{cur_time}'
RUN_STEP_RECORD_PICKLE_FILE = 'step_record/step_record_list.pkl'
# RUN_EPISODE_MOTOR_KP_PICKLE_FILE = '{policy_name}/episode_motor_kp.pkl'
RUN_EPISODE_MOTOR_KP_PICKLE_FILE = 'episode_motor_kp.pkl'


@dataclass(init=True)
class _StepTimePnt:
    step_start:float = float('inf')
    recv_obs: float = float('inf')
    inference: float = float('inf')
    set_action: float = float('inf')
    sim_step: float = float('inf')
    step_end: float = float('inf')

@dataclass(init=True)
class StepRecord:
    time_pnt: _StepTimePnt = field(default_factory=_StepTimePnt)
    obs: Obs = None
    motor_act: npt.NDArray[np.float32] = None
    ctrl_input: Dict[str, float] = None  # e.g., human operation.

class sysIDEpisodeInfo(NamedTuple):
    ep_end_time_pnt: float
    sysID_jnt_name: List[str]
    motor_kp: Dict[str, float]


__all__ = ['sysIDEpisodeInfo',
           'RUN_POLICY_LOG_FOLDER_FMT',
           'RUN_STEP_RECORD_PICKLE_FILE',
           'RUN_EPISODE_MOTOR_KP_PICKLE_FILE',
           'StepRecord']


