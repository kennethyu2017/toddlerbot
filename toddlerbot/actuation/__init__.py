from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence
from contextlib import contextmanager

@dataclass
class JointState:
    """Data class for storing joint state information"""
    # instance variables, mutable. (NamedTuple is immutable)
    time: float
    pos: float
    vel: float = 0.0
    tor: float = 0.0


class BaseController(ABC):
    """Base class for motor controllers"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def connect_to_client(self):
        pass

    @abstractmethod
    def initialize_motors(self):
        pass

    @abstractmethod
    def set_pos(self, pos: List[float]):
        pass

    @abstractmethod
    def get_motor_state(self, retries: int=0) -> Dict[int, JointState]:
        pass

    @abstractmethod
    def close_motors(self):
        pass

    @abstractmethod
    def set_kp(self, kp: Sequence[int|float]):...


    @staticmethod
    @abstractmethod
    def disable_motors(ids=None):...


    @staticmethod
    @abstractmethod
    def enable_motors(ids=None):...


    @classmethod
    @contextmanager
    def open_controller(cls, *args, **kwargs):
        # cls should be sub-class.
        assert cls is not BaseController

        controller = None
        try:
            controller = cls(args, kwargs)
            yield controller
        except (IOError,OSError) as err:
            print(f'open controller {cls.__name__} got error: {err} {type(err)=:},'
                  f'check the USE serial connections...')
        finally:
            if controller is not None:
                controller.close_motors()



from .feite_control import FeiteController, FeiteConfig
from .dynamixel_control import DynamixelController, DynamixelConfig


__all__ = ['FeiteController', 'FeiteConfig', 'DynamixelController', 'DynamixelConfig',
           'BaseController', 'JointState',]


