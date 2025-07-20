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
    temp: float = 0.0


class BaseController(ABC):
    """Base class for motor controllers"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def connect_to_client(self, usb_com_latency_timer_ms:int, timeout_ms: int):
        pass

    @abstractmethod
    def initialize_motors(self):...

    # @abstractmethod
    # used for asyncio.
    # async def send_rcv_task(self):...

    @abstractmethod
    def set_pos(self, pos: List[float]):
        pass

    @abstractmethod
    def get_motor_state(self, timeout_sec: float) -> Dict[int, JointState]:
        pass

    @abstractmethod
    def close_motors(self):
        pass

    @abstractmethod
    def set_kp(self, kp: Sequence[int|float]):...

    @abstractmethod
    def set_pos_kp(self, kp: Sequence[int|float]):...


    @abstractmethod
    def disable_motors(self,ids=None):...


    @abstractmethod
    def enable_motors(self,ids=None):...

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


