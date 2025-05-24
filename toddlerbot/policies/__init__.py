from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

from ..utils import interpolate, snake2camel

# from toddlerbot.utils.math_utils import interpolate
# from toddlerbot.utils.misc_utils import snake2camel

from .balance_pd import BalancePDPolicy
from .calibrate import CalibratePolicy
from .dp_policy import DPPolicy
from .mjx_policy import MJXPolicy
from .push_cart import PushCartPolicy
from .record import RecordPolicy
from .replay import ReplayPolicy
from .sysID import SysIDPolicy
from .teleop_follower_pd import TeleopFollowerPDPolicy
from .teleop_joystick import TeleopJoystickPolicy
from .teleop_leader import TeleopLeaderPolicy

__all__ = ['BasePolicy', 'get_policy_class', 'get_policy_names','BalancePDPolicy', 'CalibratePolicy', 'DPPolicy', 'MJXPolicy', 'PushCartPolicy',
            'RecordPolicy','ReplayPolicy', 'SysIDPolicy', 'TeleopFollowerPDPolicy', 'TeleopJoystickPolicy', 'TeleopLeaderPolicy',
           ]

# Global registry to store policy names and their corresponding classes
# TODO: move into BasePolicy as class variable.
__policy_registry: Dict[str, Type["BasePolicy"]] = {}


def get_policy_class(policy_name: str) -> Type["BasePolicy"]:
    """Retrieves the policy class associated with the given policy name.

    Args:
        policy_name (str): The name of the policy to retrieve.

    Returns:
        Type[BasePolicy]: The class of the policy corresponding to the given name.

    Raises:
        ValueError: If the policy name is not found in the policy registry.
    """
    if policy_name not in __policy_registry:
        raise ValueError(f"Unknown policy: {policy_name}")

    return __policy_registry[policy_name]


def get_policy_names() -> List[str]:
    """Retrieves a list of policy names from the policy registry.

    This function iterates over the keys in the policy registry and generates a list
    of policy names. For each key, it adds the key itself and a modified version of
    the key with the suffix '_fixed' to the list.

    Returns:
        List[str]: A list containing the original and modified policy names.
    """
    policy_names: List[str] = []
    for key in __policy_registry.keys():
        policy_names.append(key)
        policy_names.append(key + "_fixed")

    return policy_names


class BasePolicy(ABC):
    """Base class for all policies."""

    @abstractmethod
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        # control_dt: float = 0.02,    # 20ms, 50Hz.
        # prep_duration: float = 2.0,
        # n_steps_total: float = float("inf"),
    ):
        """Initializes the class with robot configuration and control parameters.

        Args:
            name (str): The name of the robot or component.
            robot (Robot): An instance of the Robot class containing robot specifications.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            control_dt (float, optional): Time interval for control updates. Defaults to 0.02.
            prep_duration (float, optional): Duration for preparation phase. Defaults to 2.0.
            n_steps_total (float, optional): Total number of control steps. Defaults to infinity.
        """
        self.name = name
        self.robot = robot
        self.init_motor_pos = init_motor_pos

        # self.control_dt = control_dt
        # self.prep_duration = prep_duration
        # self.n_steps_total = n_steps_total

        # some default values. can be overridden.
        self.control_dt: float = 0.02      # 20ms, 50Hz.
        self.prep_duration: float = 2.0
        self.n_steps_total: float = float("inf")

        self.header_name = snake2camel(name)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_active_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.joint_cfg_limits[name] for name in robot.motor_name_ordering]
        )
        indices = np.arange(robot.nu)
        motor_groups = np.array(
            [robot.joint_cfg_groups[name] for name in robot.motor_name_ordering]
        )
        joint_groups = np.array(
            [robot.joint_cfg_groups[name] for name in robot.active_joint_name_ordering]
        )
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        # self.prep_duration = 2.0
        # self.prep_time, self.prep_action = self.move(
        #     -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        # )

    # Automatic registration of subclasses
    def __init_subclass__(cls, policy_name: str = "", **kwargs):
        """Initializes a subclass and registers it with a policy name.

        Args:
            cls: the subclass.
            policy_name (str): The name of the policy to register the subclass under. If not provided, the subclass will not be registered.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        if len(policy_name) > 0:
            global __policy_registry
            __policy_registry[policy_name] = cls

    def reset(self):...

    @abstractmethod
    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:...


    # duration: total length of the motion
    # end_time: when motion should end, end time < time < duration will keep static.
    def move(
        self, *,
        time_curr: float,
        action_curr: npt.NDArray[np.float32],
        action_next: npt.NDArray[np.float32],
        duration: float,
        end_time: float = 0.0,
    )->Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Calculates the trajectory of an action over a specified duration, interpolating between current and next actions.

        Args:
            time_curr (float): The current time from which the trajectory starts.
            action_curr (npt.NDArray[np.float32]): The current action state of all motors as a NumPy array.
            action_next (npt.NDArray[np.float32]): The next action state of all motors as a NumPy array.
            duration (float): The total duration over which the action should be interpolated.
            end_time (float, optional): The duration time at the end of the duration where the action should remain constant,
                    i.e., keep action_next inside `end_time`. Defaults to 0.0.

        Returns:
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: A tuple containing the time steps and the corresponding interpolated positions.
        """
        time_seq: npt.NDArray[np.float32] = np.linspace(
            start=0,
            stop=duration,
            num=int(duration / self.control_dt),
            endpoint=False,
            dtype=np.float32,
        )


        assert action_curr.shape[0] == self.robot.nu
        # shape: ( time_seq_len, robot.nu )
        act_seq = np.zeros((len(time_seq), action_curr.shape[0]), dtype=np.float32)

        moving_dur = duration - end_time
        for i, t in enumerate(time_seq):
            if t < moving_dur:
                # TODO : us np.linspace directly...?
                act_seq[i] = interpolate(
                    p_start=action_curr,
                    p_end=action_next,
                    duration=moving_dur,
                    t=t )
            else:
                act_seq[i] = action_next   # keep action_next inside `end_time`.

        # TODO: why add control_dt?
        time_seq += time_curr + self.control_dt

        return time_seq, act_seq

