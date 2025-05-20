from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

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

