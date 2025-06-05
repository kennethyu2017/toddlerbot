from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from ...sim import ( Obs,Robot )
from ...utils import interpolate_action
from ..base_policy import BasePolicy

# This script runs the simple stand policy.


class StandPolicy(BasePolicy, policy_name="stand"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the object with a name, a robot instance, and initial motor positions.

        Args:
            name (str): The name of the object.
            robot (Robot): An instance of the Robot class.
            init_motor_pos (npt.NDArray[np.float32]): An array of initial motor positions.
        """
        super().__init__(name, robot, init_motor_pos)

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Generates the next action based on the current observation and preparation phase.

        Args:
            obs (Obs): The current observation containing the time and other relevant data.
            is_real (bool, optional): Flag indicating if the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the next action as a numpy array. If the current time is within the preparation duration, the action is interpolated based on the preparation time and action; otherwise, the default motor position is returned.
        """
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        return {}, self.default_motor_pos
