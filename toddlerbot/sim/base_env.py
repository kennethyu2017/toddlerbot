from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

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


class BaseEnv(ABC):
    """Base class for simulation environments"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    # def __init__(self, name: str):
    #     self.name = name

    def __init_subclass__(cls, env_name: str, **kwargs):
        """
        Args:
            cls: the subclass.
            env_name (str):  `real world`, `mujoco`.
        """
        super().__init_subclass__(**kwargs)
        cls._env_name = env_name

    @abstractmethod
    def set_motor_target(self, motor_angles: Dict[str, float]| npt.NDArray[np.float32]):
        pass

    @abstractmethod
    def set_motor_kps(self, motor_kps: Dict[str, float]):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_observation(self, retries:int=0) -> Obs:
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    def env_name(self):
        return self._env_name  # class variable of subclass.
