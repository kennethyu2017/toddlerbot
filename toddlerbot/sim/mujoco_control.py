from typing import Optional,Mapping
from dataclasses import dataclass

from ..utils import ( array_lib as np,
                      ArrayType)

@dataclass(init=True)
class MotorController:
    """A class for controlling the Dynamixel motors of a robot."""
    _kp: ArrayType
    _kd: ArrayType
    _tau_max: ArrayType
    _q_dot_tau_max: ArrayType
    _q_dot_max: ArrayType

    # def __init__(self, kp:ArrayType, kd:ArrayType, tau_max:ArrayType, q_dot_tau_max:ArrayType, q_dot_max:ArrayType):
    # def __init__(self, robot: Robot):
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

    # # NOTE: use kp_sim instead kp_real.
    # TODO: add setter/getter for _kp, _kd, .... not allow modify directly.

    # self._kp: List[float] = np.asarray([robot.motor_kp_sim[_n] for _n in robot.motor_name_ordering], dtype=np.float32)
    #
    # self._kd: List[float] = np.asarray([robot.motor_kd_sim[_n] for _n in robot.motor_name_ordering], dtype=np.float32)
    #
    # self._tau_max: List[float] = np.asarray([robot.motor_tau_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)
    #
    # self._q_dot_tau_max: List[float] = np.asarray([robot.motor_q_dot_tau_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)
    #
    # self._q_dot_max: List[float] = np.asarray([robot.motor_q_dot_max[_n] for _n in robot.motor_name_ordering], dtype=np.float32)

    def _set_param_helper(self, param_name: str, set_value: Mapping[int, float]):
        param_value:ArrayType = getattr(self, param_name)

        for _idx, _v in set_value.items():
            # _idx is the index of motor/actuator.
            assert 0 <= _idx < len(param_value)
            param_value[_idx] = _v

        setattr(self, param_name, param_value)

    def set_kp(self, set_value: Mapping[int, float]):
        """
        Args:
            set_value: key is the index of motor/actuator.
        """
        self._set_param_helper('_kp',set_value)

    def set_kd(self, set_value: Mapping[int, float]):
        self._set_param_helper('_kd',set_value)

    def set_tau_max(self, set_value: Mapping[int, float]):
        self._set_param_helper('_tau_max',set_value)

    def set_q_dot_tau_max(self, set_value: Mapping[int, float]):
        self._set_param_helper('_q_dot_tau_max',set_value)

    def set_q_dot_max(self, set_value: Mapping[int, float]):
        self._set_param_helper('_q_dot_max',set_value)

    @property
    def tau_max(self)->ArrayType:
        return self._tau_max

    @property
    def q_dot_tau_max(self) ->ArrayType:
        return self._q_dot_tau_max

    @property
    def q_dot_max(self)->ArrayType:
        return self._q_dot_max

    # def set_(self, set_value: Mapping[int, float]):
    #     self._set_param_helper('_',set_value)


    # @property
    # def kp(self)->ArrayType:
    #     return self._kp
    #
    # @kp.setter
    # def kp(self, value:ArrayType)->ArrayType:
    #     self._kp = value
    #
    # @property
    # def kd(self) -> ArrayType:
    #     return self._kd
    #
    # @kd.setter
    # def kd(self, value: ArrayType) -> ArrayType:
    #     self._kd = value
    #
    # @property
    # def tau_max(self) -> ArrayType:
    #     return self._tau_max
    #
    # @tau_max.setter
    # def tau_max(self, value: ArrayType) -> ArrayType:
    #     self._tau_max = value
    #
    # @property
    # def q_dot_tau_max(self) -> ArrayType:
    #     return self._q_dot_tau_max
    #
    # @q_dot_tau_max.setter
    # def q_dot_tau_max(self, value: ArrayType) -> ArrayType:
    #     self._q_dot_tau_max = value
    #
    # @property
    # def q_dot_max(self) -> ArrayType:
    #     return self._q_dot_max
    #
    # @q_dot_max.setter
    # def q_dot_max(self, value: ArrayType) -> ArrayType:
    #     self._q_dot_max = value


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
