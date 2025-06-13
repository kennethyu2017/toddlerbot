from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Mapping, OrderedDict
from collections import OrderedDict
from itertools import product

import numpy as np
import numpy.typing as npt

if __name__ == '__main__':
    from pathlib import Path
    import time as timelib
    import logging

    from toddlerbot.sim import (Obs, Robot)
    from toddlerbot.utils import (get_chirp_signal, interpolate_action, set_seed, config_logging)
    from toddlerbot.policies import sysIDEpisodeInfo,RUN_POLICY_LOG_FOLDER_FMT
    from toddlerbot.policies.base_policy import BasePolicy
    from toddlerbot.policies._module_logger import logger
    from toddlerbot.visualization import *

else:
    from ...sim import ( Obs,Robot )
    from ...utils import ( get_chirp_signal, interpolate_action, set_seed, config_logging )
    from ..base_policy import BasePolicy
    from .._module_logger import logger
    from .. import sysIDEpisodeInfo

# This script collects data for system identification of the motors.
# in seconds.
_WARM_UP_DURATION = 2.0
_CHIRP_SIGNAL_DURATION = 10.0
_CHIRP_START_FREQ = 0.1

# TODO: 3 is enough?
_CHIRP_END_FREQ = 10.

_CHIRP_DECAY_RATE = 0.1
_RESET_DURATION = 2.0


# TODO: put into yaml.
@dataclass(init=True)
class _SysIDSpecs:
    """Dataclass for system identification specifications."""

    amplitude_ratio_list: List[float]
    initial_frequency: float = _CHIRP_START_FREQ # 0.1
    final_frequency: float = _CHIRP_END_FREQ # 10.0
    decay_rate: float = _CHIRP_DECAY_RATE  # 0.1
    direction: int = 1  # {1, -1}
    kp_list: Optional[List[float]] = None

    # other accompany active joint angles, not the sysID motor's `act`.
    accompany_jnt_warm_up_angles: Optional[Dict[str, float]] = None


# TODO: put into yaml.
def _build_jnt_sysID_spec(robot_name: str)->Mapping[str, _SysIDSpecs]:

    # NOTE: the key is joint name corresponding to `robot.active_joint` name.
    # Not the motor name, but must be 1-to-1 mapping to motor name.
    specs : Mapping[str, _SysIDSpecs] | None = None

    if "sysID" in robot_name:  # for single motor sysID.
        kp_list: List[float]|None = None
        if "330" in robot_name:
            # TODO: will be divided by 128 in Dynamixel actuator's internal Position PD controller.
            kp_list = list(range(900, 2400, 300))
        elif 'sm40bl' in robot_name.casefold():
            # kp_list = list(range(17, 47, 4))  # defualt kp is `32` for SM40BL.
            kp_list = [16]
        else:
            kp_list = list(range(1500, 3600, 300))

        # single motor joint.
        specs = {
            # "joint_0": _SysIDSpecs(amplitude_ratio_list=[0.25, 0.5, 0.75], kp_list=kp_list)
            "joint_0": _SysIDSpecs(amplitude_ratio_list=[0.25], kp_list=kp_list)
        }

    else:  # for multi-links sysID.
        XC330_kp_list = [1200.0, 1500.0, 1800.0]
        # XC430_kp_list = [1800.0, 2100.0, 2400.0]
        # XM430_kp_list = [2400.0, 2700.0, 3000.0]

        # symm motor joint.
        specs = {

            # "neck_yaw_driven": _SysIDSpecs(amplitude_max=np.pi / 2),
            # "neck_pitch": _SysIDSpecs(),
            "ank_roll": _SysIDSpecs(
                amplitude_ratio_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
            ),
            "ank_pitch": _SysIDSpecs(
                amplitude_ratio_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
            ),
            # "knee": _SysIDSpecs(
            #     amplitude_ratio_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 12,
            #         "right_sho_roll": -np.pi / 12,
            #         "left_hip_roll": np.pi / 8,
            #         "right_hip_roll": np.pi / 8,
            #     },
            #     direction=-1,
            #     kp_list=XM430_kp_list,
            # ),
            # "hip_pitch": _SysIDSpecs(
            #     amplitude_ratio_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 12,
            #         "right_sho_roll": -np.pi / 12,
            #         "left_hip_roll": np.pi / 8,
            #         "right_hip_roll": np.pi / 8,
            #     },
            #     kp_list=XC430_kp_list,
            # ),
            # "hip_roll": _SysIDSpecs(
            #     amplitude_ratio_list=[0.25, 0.5, 0.75],
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            #     kp_list=XC430_kp_list,
            # ),

            # driven by gear transmission.
            "hip_yaw_driven": _SysIDSpecs(
                amplitude_ratio_list=[0.25, 0.5, 0.75],

                # active joint angles, not same thing of `motor act`.
                accompany_jnt_warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            "waist_roll": _SysIDSpecs(
                amplitude_ratio_list=[0.25, 0.5, 0.75],
                accompany_jnt_warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            "waist_yaw": _SysIDSpecs(
                amplitude_ratio_list=[0.25, 0.5, 0.75],
                accompany_jnt_warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
                kp_list=XC330_kp_list,
            ),
            # "sho_yaw_driven": _SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            # ),
            # "elbow_yaw_driven": _SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            #     direction=-1,
            # ),
            # "wrist_pitch_driven": _SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #     },
            # ),
            # "elbow_roll": _SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #         "left_sho_yaw_driven": -np.pi / 2,
            #         "right_sho_yaw_driven": -np.pi / 2,
            #     },
            # ),
            # "wrist_roll": _SysIDSpecs(
            #     warm_up_angles={
            #         "left_sho_roll": -np.pi / 6,
            #         "right_sho_roll": -np.pi / 6,
            #         "left_sho_yaw_driven": -np.pi / 2,
            #         "right_sho_yaw_driven": -np.pi / 2,
            #     },
            # ),
            # "sho_pitch": _SysIDSpecs(),
            # "sho_roll": _SysIDSpecs(),
        }

    return specs


class SysIDPolicy(BasePolicy, policy_name="sysID"):
# class SysIDFixedPolicy(BasePolicy, policy_name="sysID"):
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
            episode_motor_kp (Dict[float, Dict[str, float]]): Dictionary storing checkpoint data for joint names and their corresponding kp values.
            time_arr (npt.NDArray[np.float32]): Concatenated array of time steps for the entire process.
            action_arr (npt.NDArray[np.float32]): motor actions. Concatenated array of actions corresponding to each time step.
            n_steps_total (int): Total number of steps in the time array.
        """

        # TODO: robot.motor_id_ordering should be 1-to-1 mapping with robot.active_joint_name_ordering.
        assert len(init_motor_pos) == len(robot.motor_id_ordering) == len(robot.active_joint_name_ordering)
        super().__init__(name, robot, init_motor_pos)
        set_seed(0)

        self._start_step :bool = False

        # self.prep_duration = 2.0   # 2 sec.
        # _WARM_UP_DURATION = 2.0
        # _SIGNAL_DURATION = 10.0
        # _RESET_DURATION = 2.0

        jnt_sysID_specs = _build_jnt_sysID_spec(robot.name)

        # time_list: List[npt.NDArray[np.float32]] = []
        # action_list: List[npt.NDArray[np.float32]] = []
        # self.time_arr:npt.NDArray[np.float32] | None= None
        # self.action_arr:npt.NDArray[np.float32] | None = None

        # TODO: change key index from float time to int index value.
        # self.episode_motor_kp: OrderedDict[float, Dict[str, float]] = OrderedDict()
        # self.episode_motor_kp: List[ Tuple[float, Dict[str, float]] ] = []
        # must be ordered list.

        self.episode_info: List[sysIDEpisodeInfo] = []

        # NOTE: `act` is motor control action, e.g., target pos of motor.
        # In prep duration, make all motors back to zero.
        # In the following steps, the default angels for irrelative motors are kept `0`. easy way.
        prep_time_seq, prep_motor_act_seq = self.move(time_curr= -self.control_dt_sec,
                                                      action_curr=init_motor_pos,
                                                      action_next= np.zeros_like(init_motor_pos),
                                                      duration=self.prep_duration)

        # for whole sysID procedure.
        self._sysID_time_seq: npt.NDArray[np.float32] = prep_time_seq
        self._sysID_motor_act_seq: npt.NDArray[np.float32] = prep_motor_act_seq

        # time_list.append(prep_time)
        # action_list.append(prep_action)

        logger.info(f'{init_motor_pos=:}')

        # NOTE: guarantee only one sysID_joint per one episode. but one sysID_joint has multiple
        # episode with different kp/ampl.
        for _symm_jnt_name, _sysID_specs in jnt_sysID_specs.items():
            # joint_idx: List[int] | None = None
            sysID_jnt_name : List[str] | None = None
            sysID_jnt_dir: Mapping[str, int] | None = None

            # warm_up_pos = np.zeros_like(init_motor_pos)
            # active_jnt_warm_up_angles = np.full(shape= len(robot.active_joint_name_ordering),
            #                            fill_value=np.inf,
            #                            dtype=np.float32)

            if _symm_jnt_name in robot.active_joint_name_ordering:
                sysID_jnt_name = [_symm_jnt_name]
                # joint_idx = [robot.active_joint_name_ordering.index(joint_names[0])]
                sysID_jnt_dir = {sysID_jnt_name[0]:1}
            else:
                sysID_jnt_name = [f"left_{_symm_jnt_name}", f"right_{_symm_jnt_name}"]
                assert sysID_jnt_name[0] in self.robot.active_joint_name_ordering
                assert sysID_jnt_name[1] in self.robot.active_joint_name_ordering

                # TODO: `direction` is weired...
                sysID_jnt_dir = {sysID_jnt_name[0]: 1,
                           sysID_jnt_name[1]: _sysID_specs.direction}
                # joint_idx = [
                #     robot.active_joint_name_ordering.index(joint_names[0]),
                #     robot.active_joint_name_ordering.index(joint_names[1]),
                # ]

            logger.info(f'{_symm_jnt_name=:}, {sysID_jnt_name=:}, {sysID_jnt_dir=:}')

            # --- calc warm up action:
            mean_angle = (
                robot.joint_cfg_limits[sysID_jnt_name[0]][0]
                + robot.joint_cfg_limits[sysID_jnt_name[0]][1]
            ) / 2.
            amplitude_max = robot.joint_cfg_limits[sysID_jnt_name[0]][1] - mean_angle
            logger.info(f'{mean_angle=:}, {amplitude_max=:}')

            # warm up sysID motor:
            # active_jnt_warm_up_angles[joint_idx[0]] = mean
            # active_jnt_warm_up_angle[jnt_name[0]] = mean
            # if len(jnt_name) > 1:
            #     active_jnt_warm_up_angle[jnt_name[1]] = mean * _sysID_specs.direction

            # NOTE: assign valid values for warm_up sysID joints and other accompany active joints.
            active_jnt_warm_up_angle: OrderedDict[str, float] = OrderedDict(
                (_n, 0.) for _n in robot.active_joint_name_ordering
            )

            # warm up sysID motor:
            for _n, _d in sysID_jnt_dir.items():
                active_jnt_warm_up_angle[_n] = mean_angle * _d

            # warm up other accompany joints:
            # NOTE: `active_jnt_warm_up_angle` include not only the sysID joints, but also the
            # accompany warm-up joints.
            if _sysID_specs.accompany_jnt_warm_up_angles is not None:
                for _n, _a in _sysID_specs.accompany_jnt_warm_up_angles.items():
                    # active_jnt_warm_up_angles[robot.active_joint_name_ordering.index(_n)] = _a
                    assert _n in robot.active_joint_name_ordering
                    active_jnt_warm_up_angle[_n] = _a

            # TODO: robot.motor_id_ordering should be 1-to-1 mapping with robot.active_joint_name_ordering.
            # motor_warm_up_angles = robot.active_joint_to_motor_angles(joints_config=robot.config,
            #                                                           joint_angles= dict(zip(robot.active_joint_name_ordering, warm_up_pos)) )

            # TODO: len(active_jnt_warm_up_angles) != robot.nu......
            # motor_act_warm_up = np.zeros_like(robot.motor_name_ordering, dtype=np.float32)
            # for _k, _v in robot.active_joint_to_motor_angles(active_jnt_warm_up_angles):
            #     idx = robot.motor_name_ordering.index(_k)
            #     motor_act_warm_up[idx] = _v

            # warm_up_act = np.array(
            #     list(warm_up_motor_angles.values()), dtype=np.float32
            # )

            # --- calc chirp signal:
            # chirp_time_seq, chirp_signal_seq = get_chirp_signal(
            #     _SIGNAL_DURATION,
            #     self.control_dt,
            #     0.0,
            #     sysID_specs.initial_frequency,
            #     sysID_specs.final_frequency,
            #     amplitude_ratio * amplitude_max,
            #     sysID_specs.decay_rate,
            # )


            # if _sysID_specs.kp_list is None:
            #     for _ratio in _sysID_specs.amplitude_ratio_list:
            #         chirp_param = dict(
            #             duration=_SIGNAL_DURATION,
            #             control_dt=self.control_dt,
            #             mean=0.0,
            #             initial_frequency=_sysID_specs.initial_frequency,
            #             final_frequency=_sysID_specs.final_frequency,
            #             amplitude=_ratio * amplitude_max,
            #             decay_rate=_sysID_specs.decay_rate)
            #
            #         ep_time_seq, ep_act_seq =self.build_episode(time_curr=self._sysID_time_seq[-1],
            #                            action_curr=self._sysID_motor_act_seq[-1],
            #                            sysID_jnt_direction=active_jnt_dir,
            #                            active_jnt_warm_up_angle=active_jnt_warm_up_angle,
            #                            duration_warm_up=_WARM_UP_DURATION,
            #                            duration_reset=_RESET_DURATION,
            #                            chirp_signal_param=chirp_param)
            #
            #         self._sysID_time_seq = np.concatenate([self._sysID_time_seq, ep_time_seq], axis=0, dtype=np.float32)
            #         self._sysID_motor_act_seq = np.concatenate([self._sysID_motor_act_seq, prep_motor_act_seq], axis=0, dtype=np.float32)
            #
            #         # use kp=0.0, no feedback?
            #         # TODO: episode_motor_kp indexed by episode start time index in `_sysID_time_seq`.
            #         self.episode_motor_kp[self._sysID_time_seq[-1]] = dict(
            #             zip(active_jnt_name, [0.] * len(active_jnt_name))
            #         )
            # else:

            if _sysID_specs.kp_list is None:
                # TODO: why use 0, PD controller not work? how about use defaults?
                kp_list = [0.]
            else:
                kp_list =_sysID_specs.kp_list

            # for kp in _sysID_specs.kp_list:
            # for _kp in kp_list:
            #     for _ratio in _sysID_specs.amplitude_ratio_list:

            sysID_motor_name: List[str] = []
            for _n in sysID_jnt_name:
                sysID_motor_name.extend(self.robot.active_joint_to_motor_name[_n])

            # NOTE: in one episode, can have 1~2 sysID_joints
            for _kp, _ratio in product(kp_list, _sysID_specs.amplitude_ratio_list):
                chirp_param = dict(
                    duration=_CHIRP_SIGNAL_DURATION,
                    control_dt=self.control_dt_sec,
                    mean=0.0,
                    initial_frequency=_sysID_specs.initial_frequency,
                    final_frequency=_sysID_specs.final_frequency,
                    amplitude=_ratio * amplitude_max,
                    decay_rate=_sysID_specs.decay_rate)

                # NOTE: `active_jnt_warm_up_angle` include not only the sysID joints, but also the
                # accompany warm-up joints.
                ep_time_seq, ep_act_seq = self._build_motor_act_episode(time_curr=self._sysID_time_seq[-1],
                                                                        action_curr=self._sysID_motor_act_seq[-1],
                                                                        sysID_jnt_direction=sysID_jnt_dir,
                                                                        active_jnt_warm_up_angle=active_jnt_warm_up_angle,
                                                                        duration_warm_up=_WARM_UP_DURATION,
                                                                        duration_reset=_RESET_DURATION,
                                                                        chirp_signal_param=chirp_param)

                self._sysID_time_seq = np.concatenate([self._sysID_time_seq, ep_time_seq], axis=0,
                                                      dtype=np.float32)
                self._sysID_motor_act_seq = np.concatenate([self._sysID_motor_act_seq, ep_act_seq],   #  prep_motor_act_seq],
                                                           axis=0, dtype=np.float32)

                # motor_name: List[str] = []
                # for _n in active_jnt_name:
                #     motor_name.extend(self.robot.active_joint_to_motor_name[_n])

                # self.episode_motor_kp[self._sysID_time_seq[-1]] = { _n: _kp for _n in motor_name}
                # must be ordered list.
                self.episode_info.append(sysIDEpisodeInfo(ep_end_time_pnt=self._sysID_time_seq[-1],
                                                          sysID_jnt_name=sysID_jnt_name,
                                                          motor_kp={ _n: _kp for _n in sysID_motor_name}))

                logger.info(f'--->build episode, end time: {self._sysID_time_seq[-1]}, end act: {self._sysID_motor_act_seq[-1]}'
                            f'\n active jnt name: {sysID_jnt_name}, active jnt direction: {sysID_jnt_dir}, '
                            f'\n active jnt warm_up angle: {active_jnt_warm_up_angle}, amplitude ratio: {_ratio}'
                            f'\n sysID_time_seq shape: {self._sysID_time_seq.shape}, sysID_act_seq shape: {self._sysID_motor_act_seq.shape} '
                            f'\n kp for motors: {self.episode_info[-1]} ')

                # self.episode_motor_kp[self._sysID_time_seq[-1]] = dict(
                #     zip(active_jnt_name, [kp] * len(active_jnt_name))
                # )

        # self.time_arr = np.concatenate(time_list)
        # self.action_arr = np.concatenate(action_list)

        # override the value set in BasePolicy.__init__()
        self.n_steps_total = len(self._sysID_time_seq)
        logger.info(f'finish building all the episodes: {self.n_steps_total=:} '
                    f'sysID end time: {self._sysID_time_seq[-1]}, end act: {self._sysID_motor_act_seq[-1]}')


    # one episode: act_seq from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up.
    # NOTE: in one episode, can have 1~2 sysID_joints.
    def _build_motor_act_episode(self, *, time_curr: float,
                                action_curr: npt.NDArray[np.float32],
                                # action_warm_up: npt.NDArray[np.float32],
                                # NOTE: only for sysID active jnt, not include the accompany
                                # active joints for warm_up. we add `chirp` signal only onto sysID joints.
                                sysID_jnt_direction: Mapping[str, int],

                                # NOTE: include warm_up sysID motors and other accompany active joints.
                                # so not to add chirp signal to all joints among `active_jnt_warm_up_angle`.
                                active_jnt_warm_up_angle: OrderedDict[str, float],

                                duration_warm_up: float,
                                duration_reset: float,
                                chirp_signal_param: Dict[str, float],
                                # kp: float
                                ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        assert action_curr.shape[0] == self.robot.nu

        # at most 2 sysID joint per episode.
        assert len(sysID_jnt_direction) <= 2

        # including from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up
        # shape: ( time_seq_len, robot.nu )
        episode_time_seq: npt.NDArray[np.float32] | None = None
        episode_act_seq: npt.NDArray[np.float32] | None = None

        # --- warm up:
        # act_warm_up = np.zeros_like(self.robot.motor_name_ordering, dtype=np.float32)

        # for _n, _a in self.robot.active_joint_to_motor_angles(active_jnt_warm_up_angle):
        #     idx = self.robot.motor_name_ordering.index(_n)
        #     act_warm_up[idx] = _a

        # `active_jnt_warm_up_angle` include warm_up sysID motors and other accompany active joints.
        assert len(active_jnt_warm_up_angle) == len(self.robot.active_joint_name_ordering)
        motor_warm_up_angle = self.robot.active_joint_to_motor_angles(active_jnt_warm_up_angle)
        assert len(motor_warm_up_angle) == len(self.robot.motor_name_ordering)

        act_warm_up: npt.NDArray[np.float32] = np.asarray( [motor_warm_up_angle[_n] for _n in self.robot.motor_name_ordering],
                                  dtype=np.float32)

        if not np.allclose(act_warm_up, action_curr, 1e-06):  # self.action_arr[-1, :], 1e-6):
            warm_up_time_seq, warm_up_motor_act_seq = self.move(time_curr=time_curr,  # self.time[-1],
                                                                action_curr=action_curr,  # self.action_arr[-1, :],
                                                                action_next=act_warm_up,
                                                                duration=duration_warm_up)
            episode_time_seq = warm_up_time_seq
            episode_act_seq = warm_up_motor_act_seq

        # --- rotate joint angles by `chirp` signal.
        # NOTE: only add chirp signal onto sysID joints, not add onto `active_jnt_warm_up_angle`...
        chirp_signal_seq:npt.NDArray[np.float32]
        chirp_time_seq, chirp_signal_seq = get_chirp_signal(**chirp_signal_param)

        # if episode_time_seq is not None:
        #     chirp_time_seq += episode_time_seq[-1] + self.control_dt
        # else:
        #     chirp_time_seq += time_curr + self.control_dt

        # rotate_jnt_angle_seq = np.zeros((chirp_time_seq.shape[0], robot.nu), np.float32)
        # active_jnt_rotate_angle_seq = np.zeros((chirp_time_seq.shape[0], len(self.robot.active_joint_name_ordering), np.float32)
        # shape: ( time_seq_len, robot.nu )

        # NOTE: chirp signal only for sysID joint, not include accompany active joints.
        # sysID_jnt_chirp_angle = OrderedDict( (_n, chirp_signal_seq * _d)
        #                                      for _n,_d in sysID_jnt_direction)

        # for _n, _d in sysID_jnt_direction:
        #     sysID_jnt_chirp_angle[_n] = chirp_signal_seq * _d

        # at most 2 sysID joint per episode.
        # assert len(sysID_jnt_chirp_angle) <= 2

        # chirp_motor_act_seq = np.zeros(shape=(chirp_time_seq.shape[0], len(self.robot.motor_name_ordering)),
        #                                dtype=np.float32)

        # chirp_motor_act_seq = np.repeat(act_warm_up[np.newaxis,:], repeats=chirp_time_seq.shape[0], axis=0)

        # chirp_motor_act_seq = np.tile(act_warm_up, reps=(chirp_time_seq.shape[0],1))  # shape: ( chirp_time_seq_len, robot.nu )

        # NOTE: add chirp signal as `active joint angle`, not direct `motor act`, so
        # we must convert it to motor_angle.
        # for _n, _a in self.robot.active_joint_to_motor_angles(sysID_jnt_chirp_angle):
        #     idx = self.robot.motor_name_ordering.index(_n)
        #     # add chirp onto warm_up act.
        #     chirp_motor_act_seq[:, idx] += _a  # shape: len(chirp_time_seq)

        # construct joint angle first:   {active_jnt_name: jnt_angle_seq ... }
        active_jnt_chirp_angle_seq = OrderedDict(
            (_n, chirp_signal_seq * sysID_jnt_direction[_n] if _n in sysID_jnt_direction
                else np.zeros_like(chirp_signal_seq,dtype=np.float32) )
            for _n in self.robot.active_joint_name_ordering )

        motor_chirp_angle_seq: OrderedDict[str, npt.NDArray[np.float32]] = (
            self.robot.active_joint_to_motor_angles(active_jnt_chirp_angle_seq)
        )
        assert len(motor_chirp_angle_seq) == len(self.robot.motor_name_ordering)

        # shape: (robot.nu, len(chirp_time_seq) ) -> shape: (len(chirp_time_seq), robot.nu)
        chirp_motor_act_seq: npt.NDArray[np.float32] = np.asarray(
            [motor_chirp_angle_seq[_n] for _n in self.robot.motor_name_ordering],
            dtype=np.float32).transpose()

        assert chirp_motor_act_seq.shape == ( len(chirp_time_seq), self.robot.nu )

        logger.info(f' {chirp_motor_act_seq.shape=:} ')

        # chirp_motor_act_seq = chirp_motor_act_seq.transpose()

        # add chirp onto warm_up act.
        chirp_motor_act_seq[:] += act_warm_up  # shape: ( len(chirp_time_seq), robot.nu )

        # NOTE: add chirp signal as `active joint angle`, not `motor act`.
        # rotate_jnt_angle_seq[:, joint_idx[0]] = signal_seq
        # if len(joint_idx) > 1:
        #     rotate_jnt_angle_seq[:, joint_idx[1]] = signal_seq * sysID_specs.direction

        # rotate_act_seq = np.zeros_like(rotate_jnt_angle_seq)
        # for _t_idx, _jnt_angle in enumerate(rotate_jnt_angle_seq):
        #     rotate_motor_angles = robot.active_joint_to_motor_angles(
        #         dict(zip(robot.active_joint_name_ordering, _jnt_angle))
        #     )
        #     signal_action = np.array(
        #         list(rotate_motor_angles.values()), dtype=np.float32
        #     )
        #     # add signal onto warm_up pos.
        #     rotate_act_seq[_t_idx] = signal_action + action_warm_up

        if episode_time_seq is not None:
            chirp_time_seq += episode_time_seq[-1] + self.control_dt_sec
            episode_time_seq = np.concatenate([episode_time_seq, chirp_time_seq], axis=0, dtype=np.float32)
            episode_act_seq = np.concatenate([episode_act_seq, chirp_motor_act_seq], axis=0, dtype=np.float32)
        else:
            chirp_time_seq += time_curr + self.control_dt_sec
            episode_time_seq = chirp_time_seq
            episode_act_seq = chirp_motor_act_seq

        # --- reset to warm up:
        reset_time_seq, motor_reset_act_seq = self.move(time_curr=episode_time_seq[-1],
                                                        action_curr=episode_act_seq[-1, :],
                                                        action_next=act_warm_up,
                                                        duration=duration_reset,
                                                        end_time=0.5)

        # NOTE: already reset_time_seq += episode_time_seq[-1] + self.control_dt in self.move().
        episode_time_seq = np.concatenate([episode_time_seq, reset_time_seq], axis=0, dtype=np.float32)
        episode_act_seq = np.concatenate([episode_act_seq, motor_reset_act_seq], axis=0, dtype=np.float32)

        # self.episode_motor_kp[self.time[-1]] = dict(
        #     zip(joint_names, [kp] * len(joint_names))
        # )

        return episode_time_seq, episode_act_seq

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
            interpolate_action(obs.time, self._sysID_time_seq, self._sysID_motor_act_seq)
        )

        # TODO: move to other place.
        # # check at 1st step:
        # if not self._start_step:
        #     choice:str = input(f'===> Pls confirm whether start step: current pos (normalized): {obs.motor_pos} ,'
        #                    f'action target pos: {action},'
        #                    f'check whether the load box position is safe, and confirm to action [y/n] : ')
        #     if choice.casefold() == 'n':
        #         exit(f'Abort policy step....')
        #         # raise AssertionError(f'abort policy step.')
        #
        #     self._start_step = True

        return {}, action


    # @property
    # def n_steps_total(self):
    #     return len(self._sysID_time_seq)


def get_ep_trajectory_stat(policy:SysIDPolicy)\
        ->Dict[str,npt.NDArray[np.float32]]:

    time_diff = np.diff(policy._sysID_time_seq, n=1, axis=0)
    # calc vel
    act_diff = np.diff(policy._sysID_motor_act_seq, n=1, axis=0)
    # rad/s
    target_vel = (act_diff.transpose() / time_diff).transpose()
    # rpm
    target_rpm = target_vel * 60/(2*3.14)

    # calc acc
    vel_diff = np.diff(target_vel, n=1, axis=0)
    # rad/s**2
    target_acc = (vel_diff.transpose() / time_diff[1:]).transpose()
    # round per s**2
    target_acc_rps2 = target_acc/(2*np.pi)

    logger.info(f'--- episode trajectory vel / acc stats: ---')
    logger.info(f'chirp_max_freq:{_CHIRP_END_FREQ}Hz {act_diff.shape=:} {time_diff.shape=:} {target_vel.shape=:} {target_rpm.shape=:} {target_acc.shape}')
    logger.warning(f'{max(act_diff)=:} {max(time_diff)=:} {max(target_vel)=:} rad/s '
                   f'\n{max(target_rpm)=:} {max(target_acc)=:} rad/s**2 {max(target_acc_rps2)=:} round/s**2')

    return   { 'target_acc_arr': target_acc.astype(np.float32),
               'acc_time_seq': policy._sysID_time_seq[2:],
               'target_vel_arr': target_rpm.astype(np.float32),
               'vel_time_seq': policy._sysID_time_seq[1:]
               }


def _test_main():

    config_logging(root_logger_level=logging.INFO, root_handler_level=logging.NOTSET,
                   # root_fmt='--- {levelname} - module:{module} - func:{funcName} ---> \n{message}',
                   root_fmt='--- {levelname} - module:{module} ---> \n{message}',
                   root_date_fmt='%Y-%m-%d %H:%M:%S',
                   # log_file='/tmp/toddler/imitate_episode.log',
                   log_file=None,
                   module_logger_config={'policies':logging.INFO,
                                         })

    # use root logger for __main__.
    # logger = logging.getLogger('root')

    robot = Robot('sysID_SM40BL')
    # like normalized value.
    init_motor_pos = np.zeros_like(robot.motor_id_ordering, dtype=np.float32)
    policy = SysIDPolicy('sysID', robot=robot,init_motor_pos=init_motor_pos )

    stat_dict:Dict[str,npt.NDArray[np.float32]] = get_ep_trajectory_stat(policy=policy)

    target_acc_arr: npt.NDArray[np.float32] = stat_dict['target_acc_arr']
    acc_time_seq: npt.NDArray[np.float32] = stat_dict['acc_time_seq']
    target_vel_arr: npt.NDArray[np.float32] = stat_dict['target_vel_arr']
    vel_time_seq: npt.NDArray[np.float32] = stat_dict['vel_time_seq']

    target_acc_dict: OrderedDict[str, np.ndarray[np.float32]] = OrderedDict()
    acc_time_dict: OrderedDict[str, np.ndarray[np.float32]] = OrderedDict()

    target_vel_dict: OrderedDict[str, np.ndarray[np.float32]] = OrderedDict()
    vel_time_dict: OrderedDict[str, np.ndarray[np.float32]] = OrderedDict()

    # should be (time_seq_len-1/-2, num_of_motors)
    assert target_acc_arr.shape[1] == len(robot.motor_name_ordering)
    assert target_vel_arr.shape[1] == len(robot.motor_name_ordering)

    for _x, _n in enumerate(robot.motor_name_ordering):
        target_acc_dict[_n] = target_acc_arr[:, _x]
        acc_time_dict[_n] = acc_time_seq
        target_vel_dict[_n] = target_vel_arr[:, _x]
        vel_time_dict[_n] = vel_time_seq

    exp_folder = Path(RUN_POLICY_LOG_FOLDER_FMT.format(robot_name=robot.name,
                                                       policy_name=policy.name,
                                                       env_name='test_episode',
                                                       cur_time=timelib.strftime("%Y%m%d_%H%M%S")))
    plot_dir = exp_folder / 'episode_trajectory_plot'
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)


    TODO: plot target pos....

    plot_joint_tracking_single(
        time_seq_dict=vel_time_dict,
        joint_data_dict=target_vel_dict,
        save_path=plot_dir.resolve().__str__(),
        x_label="Time (s)",
        y_label="Target Vel (RPM)",
        file_name = "target_vel_trajectory",
        set_ylim=False,
    )

    plot_joint_tracking_single(
        time_seq_dict=acc_time_dict,
        joint_data_dict=target_acc_dict,
        save_path=plot_dir.resolve().__str__(),
        x_label="Time (s)",
        y_label="Target Acc (RP/S**2)",
        file_name="target_acc_trajectory",
        set_ylim=False,
    )

if __name__ == '__main__':
    _test_main()
