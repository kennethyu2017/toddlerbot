from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from typing import Optional

@dataclass(init=True)
class Obs:
    """Observation data structure"""

    time: float = np.inf
    motor_pos: Optional[npt.NDArray[np.float32]] = None
    motor_vel: Optional[npt.NDArray[np.float32]] = None
    motor_tor: Optional[npt.NDArray[np.float32]] = None
    lin_vel: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    ang_vel: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    pos: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    euler: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    # only Mujoco can fill qpos->joint_pos , qvel->joint_ve. `real_world` only reads motor_pos.
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    joint_vel: Optional[npt.NDArray[np.float32]] = None

    # def __init__(self, nu:int):
    #     self.time = np.inf
    #     self.motor_pos = np.full(shape=nu, fill_value=np.inf,dtype=np.float32)
    #     self.motor_vel = self.motor_pos.copy()
    #     self.motor_tor = self.motor_pos.copy()

from .base_env import BaseEnv
from .mujoco_control import MotorController,PositionController
from .robot import Robot
from .mujoco_sim import MuJoCoSim
from .real_world import RealWorld

__all__ = ['MotorController',
           'MuJoCoSim',
           'Robot',
           'Obs',
           'BaseEnv',
           'PositionController',
           'RealWorld'
          ]

