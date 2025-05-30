from typing import Optional,List

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np

class MotorController:
    """A class for controlling the Dynamixel motors of a robot."""

    def __init__(self, robot: Robot):
        """Initializes the control parameters for a robot's joints using attributes specific to "dynamixel" type actuators.

        Args:
            robot (Robot): An instance of the Robot class from which joint attributes are retrieved.
        """
        # TODO: if motor is dynamixel, kp_sim is divided by 128.
        # self.kp = np.array(robot.get_joint_config_attrs("type", "dynamixel", "kp_sim"))
        # self.kd = np.array(robot.get_joint_config_attrs("type", "dynamixel", "kd_sim"))
        # self.tau_max = np.array(robot.get_joint_config_attrs("type", "dynamixel", "tau_max"))
        # self.q_dot_tau_max = np.array(
        #     robot.get_joint_config_attrs("type", "dynamixel", "q_dot_tau_max")
        # )
        # self.q_dot_max = np.array(
        #     robot.get_joint_config_attrs("type", "dynamixel", "q_dot_max")
        # )

        # NOTE: use kp_sim instead kp_real.

        TODO: add setter/getter for _kp, _kd, .... not allow modify directly.

        self._kp: List[float] = np.asarray([robot.motor_kp_sim[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

        self._kd: List[float] = np.asarray([robot.motor_kd_sim[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

        self._tau_max: List[float] = np.asarray([robot.motor_tau_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

        self._q_dot_tau_max: List[float] = np.asarray([robot.motor_q_dot_tau_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

        self._q_dot_max: List[float] = np.asarray([robot.motor_q_dot_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

    def step(
        self,
        q: ArrayType,
        q_dot: ArrayType,
        q_hat: ArrayType,
        kp: Optional[ArrayType] = None,
        kd: Optional[ArrayType] = None,
        tau_max: Optional[ArrayType] = None,
        q_dot_tau_max: Optional[ArrayType] = None,
        q_dot_max: Optional[ArrayType] = None,
    )->ArrayType:
        """Computes the clamped torque for a control step based on position, velocity, and desired acceleration.

        Args:
            q (ArrayType): Current position array.
            q_dot (ArrayType): Current velocity array.
            #a (ArrayType): Desired acceleration array.
            q_hat (ArrayType): Target position array.
            kp (Optional[ArrayType]): Proportional gain array. Defaults to self.kp.
            kd (Optional[ArrayType]): Derivative gain array. Defaults to self.kd.
            tau_max (Optional[ArrayType]): Maximum torque array. Defaults to self.tau_max.
            q_dot_tau_max (Optional[ArrayType]): Velocity threshold for maximum torque. Defaults to self.q_dot_tau_max.
            q_dot_max (Optional[ArrayType]): Maximum velocity array. Defaults to self.q_dot_max.

        Returns:
            ArrayType: Clamped torque array based on the computed control law and constraints.
        """
        if kp is None:
            kp = self._kp

        if kd is None:
            kd = self._kd

        if tau_max is None:
            tau_max = self._tau_max

        if q_dot_tau_max is None:
            q_dot_tau_max = self._q_dot_tau_max

        if q_dot_max is None:
            q_dot_max = self._q_dot_max

        # error = a - q
        error = q_hat - q
        tau_m = kp * error - kd * q_dot

        abs_q_dot = np.abs(q_dot)

        # Apply vectorized conditions using np.where
        tau_limit = np.where(
            abs_q_dot <= q_dot_tau_max,  # Condition 1
            tau_max,  # Value when condition 1 is True
            np.where(
                abs_q_dot <= q_dot_max,  # Condition 2
                tau_max / (q_dot_tau_max - q_dot_max) * (abs_q_dot - q_dot_tau_max)
                + tau_max,  # Value when condition 2 is True
                # np.zeros_like(tau_m),  # Value when all conditions are False
                0.
            ),
        )

        tau_m_clamped = np.clip(tau_m, -tau_limit, tau_limit)

        return tau_m_clamped


class PositionController:
    """A class for controlling the position of a robot's joints."""

    def step(self, q: ArrayType, q_dot: ArrayType, a: ArrayType):
        """Advances the system state by one time step using the provided acceleration.

        Args:
            q (ArrayType): The current state vector of the system.
            q_dot (ArrayType): The current velocity vector of the system.
            a (ArrayType): The acceleration vector to be applied.

        Returns:
            ArrayType: The acceleration vector `a`.
        """
        return a
