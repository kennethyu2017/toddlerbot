from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Mapping, OrderedDict

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_chirp_signal, interpolate_action
from toddlerbot.utils.misc_utils import set_seed


# This script collects data for system identification of the motors.


@dataclass
class SysIDSpecs:
    """Dataclass for system identification specifications."""

    amplitude_list: List[float]
    initial_frequency: float = 0.1
    final_frequency: float = 10.0
    decay_rate: float = 0.1
    direction: float = 1  # 1, -1
    kp_list: Optional[List[float]] = None

    # active joint angles, not same thing of `motor act`.
    warm_up_angles: Optional[Dict[str, float]] = None


def _build_jnt_sysID_spec(robot_name: str)->Mapping[str, SysIDSpecs]:

    # NOTE: the key is joint name corresponding to `robot.active_joint` name.
    # not motor name, but must be 1-to-1 mapping to motor name.
    specs : Mapping[str, SysIDSpecs] | None = None

    if "sysID" in robot_name:  # for single motor sysID.
        kp_list: List[float] = []
        if "330" in robot_name:
            # will be divided by 128 in Dynamixel actuator's internal Position PD controller.
            kp_list = list(range(900, 2400, 300))
        elif 'sm40bl' in robot_name.casefold():
            kp_list = list(range(7, 40, 5))  # defualt kp is `32` for SM40BL.
        else:
            kp_list = list(range(1500, 3600, 300))

        # single motor joint.
        specs = {
            "joint_0": SysIDSpecs(amplitude_list=[0.25, 0.5, 0.75], kp_list=kp_list)
        }

    else:  # for multi-links sysID.
        XC330_kp_list = [1200.0, 1500.0, 1800.0]
        # XC430_kp_list = [1800.0, 2100.0, 2400.0]
        # XM430_kp_list = [2400.0, 2700.0, 3000.0]

        # symm motor joint.
        specs = {

            # "neck_yaw_driven": SysIDSpecs(amplitude_max=np.pi / 2),
            # "neck_pitch": SysIDSpecs(),
            "ank_roll": SysIDSpecs(
                amplitude_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
            ),
            "ank_pitch": SysIDSpecs(
                amplitude_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
            ),
            # "knee": SysIDSpecs(
            #     amplitude_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 12,
            #         "right_sho_roll": -np.pi / 12,
            #         "left_hip_roll": np.pi / 8,
            #         "right_hip_roll": np.pi / 8,
            #     },
            #     direction=-1,
            #     kp_list=XM430_kp_list,
            # ),
            # "hip_pitch": SysIDSpecs(
            #     amplitude_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 12,
            #         "right_sho_roll": -np.pi / 12,
            #         "left_hip_roll": np.pi / 8,
            #         "right_hip_roll": np.pi / 8,
            #     },
            #     kp_list=XC430_kp_list,
            # ),
            # "hip_roll": SysIDSpecs(
            #     amplitude_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            #     kp_list=XC430_kp_list,
            # ),

            # driven by gear transmission.
            "hip_yaw_driven": SysIDSpecs(
                amplitude_list=[0.25, 0.5, 0.75],

                # active joint angles, not same thing of `motor act`.
                warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            "waist_roll": SysIDSpecs(
                amplitude_list=[0.25, 0.5, 0.75],
                warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            "waist_yaw": SysIDSpecs(
                amplitude_list=[0.25, 0.5, 0.75],
                warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            # "sho_yaw_driven": SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            # ),
            # "elbow_yaw_driven": SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            # ),
            # "wrist_pitch_driven": SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            # ),
            # "elbow_roll": SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #         "left_sho_yaw_driven": -np.pi / 2,
            #         "right_sho_yaw_driven": -np.pi / 2,
            #     },
            # ),
            # "wrist_roll": SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #         "left_sho_yaw_driven": -np.pi / 2,
            #         "right_sho_yaw_driven": -np.pi / 2,
            #     },
            # ),
            # "sho_pitch": SysIDSpecs(),
            # "sho_roll": SysIDSpecs(),
        }

    return specs


def _build_episode(*, time_curr:float,
                   action_curr: npt.NDArray[np.float32],
                   action_warm_up: npt.NDArray[np.float32], duration_warm_up:float,
                   chirp_signal_param: Dict[str,float],
                   #kp: float
                   ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # including from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up
    episode_time_seq: npt.NDArray[np.float32] | None = None
    episode_act_seq: npt.NDArray[np.float32] | None = None

    # --- warm up:
    if not np.allclose(action_warm_up, action_curr,1e-06):   #  self.action_arr[-1, :], 1e-6):
        warm_up_time_seq, warm_up_act_seq = self.move(time_curr=time_curr, # self.time[-1],
                                                 action_curr=action_curr,  #self.action_arr[-1, :],
                                                 action_next=action_warm_up,
                                                 duration=duration_warm_up)

        episode_time_seq = warm_up_time_seq
        episode_act_seq = warm_up_act_seq

    # --- rotate joint angles by `chirp` signal:
    chirp_time_seq, chirp_signal_seq = get_chirp_signal(**chirp_signal_param)

    # if episode_time_seq is not None:
    #     chirp_time_seq += episode_time_seq[-1] + self.control_dt
    # else:
    #     chirp_time_seq += time_curr + self.control_dt

    rotate_jnt_angle_seq = np.zeros((chirp_signal_seq.shape[0], robot.nu), np.float32)

    rotate_jnt_angle_seq[:, joint_idx[0]] = signal_seq
    if len(joint_idx) > 1:
        rotate_jnt_angle_seq[:, joint_idx[1]] = signal_seq * sysID_specs.direction

    rotate_act_seq = np.zeros_like(rotate_jnt_angle_seq)
    for _t_idx, _jnt_angle in enumerate(rotate_motor_angle_seq):
        rotate_motor_angles = robot.active_joint_to_motor_angles(
            dict(zip(robot.active_joint_name_ordering, _jnt_angle))
        )
        signal_action = np.array(
            list(rotate_motor_angles.values()), dtype=np.float32
        )
        # add signal onto warm_up pos.
        rotate_act_seq[_t_idx] = signal_action + action_warm_up

    if episode_time_seq is not None:
        chirp_time_seq += episode_time_seq[-1] + self.control_dt
        episode_time_seq = np.concatenate([episode_time_seq, chirp_time_seq], axis=0, dtype=np.float32)
        episode_act_seq = np.concatenate([episode_act_seq, rotate_act_seq], axis=0, dtype=np.float32)
    else:
        chirp_time_seq += time_curr + self.control_dt
        episode_time_seq = chirp_time_seq
        episode_act_seq = rotate_act_seq

    # --- reset to warm up:
    reset_time_seq, reset_act_seq = self.move(time_curr=episode_time_seq[-1],
                                         action_curr=episode_act_seq[-1, :],
                                         action_next=action_warm_up,
                                         duration=reset_duration,
                                         end_time=0.5)

    # NOTE: already reset_time_seq += episode_time_seq[-1] + self.control_dt in self.move().
    episode_time_seq = np.concatenate([episode_time_seq, reset_time_seq], axis=0, dtype=np.float32)
    episode_act_seq = np.concatenate([episode_act_seq, reset_act_seq], axis=0, dtype=np.float32)

    # self.ckpt_dict[self.time[-1]] = dict(
    #     zip(joint_names, [kp] * len(joint_names))
    # )

    return episode_time_seq, episode_act_seq


class SysIDFixedPolicy(BasePolicy, policy_name="sysID"):
    """System identification policy for the toddlerbot robot."""

    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the class with specified parameters and sets up system identification specifications for robot joints.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object containing joint and motor information.
            init_motor_pos (npt.NDArray[np.float32]): observed Initial motor positions as a NumPy array.

        Attributes:
            prep_duration (float): Duration for preparation phase.
            ckpt_dict (Dict[float, Dict[str, float]]): Dictionary storing checkpoint data for joint names and their corresponding kp values.
            time_arr (npt.NDArray[np.float32]): Concatenated array of time steps for the entire process.
            action_arr (npt.NDArray[np.float32]): motor actions. Concatenated array of actions corresponding to each time step.
            n_steps_total (int): Total number of steps in the time array.
        """

        # TODO: robot.motor_id_ordering should be 1-to-1 mapping with robot.active_joint_name_ordering.
        assert len(init_motor_pos) == len(robot.motor_id_ordering) == len(robot.active_joint_name_ordering)
        super().__init__(name, robot, init_motor_pos)
        set_seed(0)

        self.prep_duration = 2.0   # 2 sec.
        warm_up_duration = 2.0
        signal_duraion = 10.0
        reset_duration = 2.0

        jnt_sysID_specs = _build_jnt_sysID_spec(robot.name)

        # time_list: List[npt.NDArray[np.float32]] = []
        # action_list: List[npt.NDArray[np.float32]] = []
        # self.time_arr:npt.NDArray[np.float32] | None= None
        # self.action_arr:npt.NDArray[np.float32] | None = None

        self.ckpt_dict: Dict[float, Dict[str, float]] = {}

        # NOTE: `act` is motor control action, e.g., target pos of motor.
        # In prep duration, make all motors back to zero.
        # In the following steps, the default angels for irrelative motors are kept `0`. easy way.
        prep_time_seq, prep_motor_act_seq = self.move(time_curr= -self.control_dt,
                                           action_curr=init_motor_pos,
                                           action_next= np.zeros_like(init_motor_pos),
                                           duration=self.prep_duration )

        # for whole sysID procedure.
        self.sysID_time_seq: npt.NDArray[np.float32] = prep_time_seq
        self.sysID_motor_act_seq: npt.NDArray[np.float32] = prep_motor_act_seq

        # time_list.append(prep_time)
        # action_list.append(prep_action)

        for symm_joint_name, sysID_specs in jnt_sysID_specs.items():
            # joint_idx: List[int] | None = None
            joint_names : List[str] | None = None
            # warm_up_pos = np.zeros_like(init_motor_pos)
            active_jnt_warm_up_angles: OrderedDict[str, float] = {}
            # active_jnt_warm_up_angles = np.full(shape= len(robot.active_joint_name_ordering),
            #                            fill_value=np.inf,
            #                            dtype=np.float32)

            if symm_joint_name in robot.active_joint_name_ordering:
                joint_names = [symm_joint_name]
                # joint_idx = [robot.active_joint_name_ordering.index(joint_names[0])]
            else:
                joint_names = [f"left_{symm_joint_name}", f"right_{symm_joint_name}"]
                # joint_idx = [
                #     robot.active_joint_name_ordering.index(joint_names[0]),
                #     robot.active_joint_name_ordering.index(joint_names[1]),
                # ]

            # --- calc warm up action:
            mean = (
                robot.joint_cfg_limits[joint_names[0]][0]
                + robot.joint_cfg_limits[joint_names[0]][1]
            ) / 2.
            amplitude_max = robot.joint_cfg_limits[joint_names[0]][1] - mean

            # active_jnt_warm_up_angles[joint_idx[0]] = mean
            active_jnt_warm_up_angles[joint_names[0]] = mean
            if len(joint_names) > 1:
                active_jnt_warm_up_angles[joint_names[1]] = mean * sysID_specs.direction

            if sysID_specs.warm_up_angles is not None:
                for _n, _a in sysID_specs.warm_up_angles.items():
                    # active_jnt_warm_up_angles[robot.active_joint_name_ordering.index(_n)] = _a
                    assert _n in robot.active_joint_name_ordering
                    active_jnt_warm_up_angles[_n] = _a

            # TODO: robot.motor_id_ordering should be 1-to-1 mapping with robot.active_joint_name_ordering.
            # motor_warm_up_angles = robot.active_joint_to_motor_angles(joints_config=robot.config,
            #                                                           joint_angles= dict(zip(robot.active_joint_name_ordering, warm_up_pos)) )


            TODO: len(active_jnt_warm_up_angles) != nu......
            TODO: len(active_jnt_warm_up_angles) != nu......
            


            motor_warm_up_angles = robot.active_joint_to_motor_angles(active_jnt_warm_up_angles)
            warm_up_act = np.array(
                list(warm_up_motor_angles.values()), dtype=np.float32
            )

            # --- calc chirp signal:
            # chirp_time_seq, chirp_signal_seq = get_chirp_signal(
            #     signal_duraion,
            #     self.control_dt,
            #     0.0,
            #     sysID_specs.initial_frequency,
            #     sysID_specs.final_frequency,
            #     amplitude_ratio * amplitude_max,
            #     sysID_specs.decay_rate,
            # )

            if sysID_specs.kp_list is None:
                for amplitude_ratio in sysID_specs.amplitude_list:
                    chirp_signal_params = dict(
                        duration=signal_duraion,
                        control_dt=self.control_dt,
                        mean=0.0,
                        initial_frequency=sysID_specs.initial_frequency,
                        final_frequency=sysID_specs.final_frequency,
                        amplitude=amplitude_ratio * amplitude_max,
                        decay_rate=sysID_specs.decay_rate)

                    build_episode(amplitude_ratio, 0.0)
                    self.ckpt_dict[self.time[-1]] = dict(
                        zip(joint_names, [kp] * len(joint_names))
                    )
            else:
                # TODO: use permutation instead.
                for kp in sysID_specs.kp_list:
                    for amplitude_ratio in sysID_specs.amplitude_list:
                        chirp_signal_params = dict(
                            duration=signal_duraion,
                            control_dt=self.control_dt,
                            mean=0.0,
                            initial_frequency=sysID_specs.initial_frequency,
                            final_frequency=sysID_specs.final_frequency,
                            amplitude=amplitude_ratio * amplitude_max,
                            decay_rate=sysID_specs.decay_rate)
                        build_episode(amplitude_ratio, kp)
                        self.ckpt_dict[self.time[-1]] = dict(
                            zip(joint_names, [kp] * len(joint_names))
                        )

        # self.time_arr = np.concatenate(time_list)
        # self.action_arr = np.concatenate(action_list)
        self.n_steps_total = len(self.time_arr)

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the environment by interpolating an action based on the given observation time.

        Args:
            obs (Obs): The observation containing the current time.
            is_real (bool, optional): Flag indicating whether the step is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the interpolated action as a NumPy array.
        """
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        return {}, action
