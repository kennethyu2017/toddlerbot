from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from . import Obs

class BaseEnv(ABC):
    """Base class for simulation environments"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

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
