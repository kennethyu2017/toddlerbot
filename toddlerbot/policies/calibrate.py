from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from collections import OrderedDict

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action

# This script ensures a more accurate zero-point (default_pos) calibration
# by running a PID loop to control the robot's torso pitch to be `0`.
class CalibratePolicy(BasePolicy, policy_name="calibrate"):
    """Policy for calibrating zero point with the robot's torso pitch."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        kp: float = 0.1,
        kd: float = 0.01,
        ki: float = 0.2,
    ):
        """Initializes the controller with specified parameters and robot configuration.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            kp (float, optional): Proportional gain for the controller. Defaults to 0.1.
            kd (float, optional): Derivative gain for the controller. Defaults to 0.01.
            ki (float, optional): Integral gain for the controller. Defaults to 0.2.
        """

        #NOTE: init_motor_pos is env.get_observation(1).motor_pos.
        super().__init__(name, robot, init_motor_pos)

        # self.default_motor_pos = np.array(
        #     list(robot.default_motor_angles.values()), dtype=np.float32
        # )
        self.default_motor_pos: npt.NDArray[np.float32] = np.asarray(
            [ robot.default_motor_angles[_n] for _n in robot.motor_name_ordering ],
            dtype=np.float32
        )

        # self.default_joint_pos = np.array(
        #     list(robot.default_active_joint_angles.values()), dtype=np.float32
        # )
        self.default_joint_pos = np.asarray(
            [robot.default_active_joint_angles[_n] for _n in robot.active_joint_name_ordering],
            dtype=np.float32
        )

        leg_pitch_joint_names = [
            "left_hip_pitch",
            "left_knee",
            "left_ank_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ank_pitch",
        ]
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.active_joint_name_ordering.index(joint_name)
                for joint_name in leg_pitch_joint_names
            ]
        )
        self.leg_pitch_joint_signs = np.array([-1, -1, 1, 1, 1, -1], dtype=np.float32)

        # PD controller parameters
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Initialize integral error
        self.integral_error = 0.0

        self.prep_time_seq, self.prep_motor_act_seq = self.move(
            time_curr=-self.control_dt,
            action_curr=init_motor_pos,
            action_next=self.default_motor_pos,
            duration=self.prep_duration
        )

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step to maintain the torso pitch at zero using a PD+I controller.

        Args:
            obs (Obs): The current observation containing state information such as time, Euler angles, and angular velocities.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and an array of motor target angles.
        """
        # Preparation phase
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time_seq, self.prep_motor_act_seq)
            )
            return {}, action

        # PD+I controller to maintain torso pitch at 0
        error = obs.euler[1] + 0.05  # 0.05 cancels some backlash
        error_derivative = obs.ang_vel[1]

        # Update integral error (with a basic anti-windup mechanism)
        self.integral_error += error * self.control_dt
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0)  # Anti-windup

        # PID controller output
        ctrl = (
            self.kp * error + self.ki * self.integral_error - self.kd * error_derivative
        )

        # Update joint positions based on the PID controller command.
        # so the motor_target is slightly different against default joint_pos, and after the running loop of
        # Calibrate policy, we update this `bias` into joint_motor_mapping.json` `init_pos`.
        joint_pos = self.default_joint_pos.copy()
        # NOTE: Integration controller is used, so the joint_pos even when idea `Stand` pose is reached,
        # joint_pos contains the `bias`.
        joint_pos[self.leg_pitch_joint_indicies] += self.leg_pitch_joint_signs * ctrl

        # Convert joint positions to motor angles
        motor_angles = self.robot.active_joint_to_motor_angles(
            OrderedDict(zip(self.robot.active_joint_name_ordering, joint_pos))
        )
        motor_target = np.asarray([ motor_angles[_n] for _n in self.robot.motor_name_ordering],
                                  dtype=np.float32 )
        # motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        return {}, motor_target
