from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Mapping, OrderedDict, NamedTuple
from collections import OrderedDict
from itertools import product

import numpy as np
import numpy.typing as npt


from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_chirp_signal, interpolate_action
from toddlerbot.utils.misc_utils import set_seed

from ._module_logger import logger


# This script collects data for system identification of the motors.

_WARM_UP_DURATION = 2.0
_SIGNAL_DURATION = 10.0
_RESET_DURATION = 2.0


# TODO: put into yaml.
@dataclass(init=True)
class _SysIDSpecs:
    """Dataclass for system identification specifications."""

    amplitude_ratio_list: List[float]
    initial_frequency: float = 0.1
    final_frequency: float = 10.0
    decay_rate: float = 0.1
    direction: int = 1  # 1, -1
    kp_list: Optional[List[float]] = None

    # other accompany active joint angles, not the sysID motor's `act`.
    accompany_jnt_warm_up_angles: Optional[Dict[str, float]] = None


# TODO: put into yaml.
def _build_jnt_sysID_spec(robot_name: str)->Mapping[str, _SysIDSpecs]:

    # NOTE: the key is joint name corresponding to `robot.active_joint` name.
    # not motor name, but must be 1-to-1 mapping to motor name.
    specs : Mapping[str, _SysIDSpecs] | None = None

    if "sysID" in robot_name:  # for single motor sysID.
        kp_list: List[float]|None = None
        if "330" in robot_name:
            # will be divided by 128 in Dynamixel actuator's internal Position PD controller.
            kp_list = list(range(900, 2400, 300))
        elif 'sm40bl' in robot_name.casefold():
            kp_list = list(range(7, 40, 5))  # defualt kp is `32` for SM40BL.
        else:
            kp_list = list(range(1500, 3600, 300))

        # single motor joint.
        specs = {
            "joint_0": _SysIDSpecs(amplitude_ratio_list=[0.25, 0.5, 0.75], kp_list=kp_list)
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


class _EpisodeKp(NamedTuple):
    ep_end_time_pnt: float
    motor_kp: Dict[str, float]


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
        self.episode_motor_kp: List[_EpisodeKp] = []

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

        logger.info(f'{init_motor_pos=:}')

        for _symm_jnt_name, _sysID_specs in jnt_sysID_specs.items():
            # joint_idx: List[int] | None = None
            active_jnt_name : List[str] | None = None
            active_jnt_dir: Mapping[str, int] | None = None

            # warm_up_pos = np.zeros_like(init_motor_pos)
            # include warm_up sysID motors and other accompany active joints.
            active_jnt_warm_up_angle: OrderedDict[str, float] = OrderedDict()

            # active_jnt_warm_up_angles = np.full(shape= len(robot.active_joint_name_ordering),
            #                            fill_value=np.inf,
            #                            dtype=np.float32)

            if _symm_jnt_name in robot.active_joint_name_ordering:
                active_jnt_name = [_symm_jnt_name]
                # joint_idx = [robot.active_joint_name_ordering.index(joint_names[0])]
                active_jnt_dir = {active_jnt_name[0]:1}
            else:
                active_jnt_name = [f"left_{_symm_jnt_name}", f"right_{_symm_jnt_name}"]
                assert active_jnt_name[0] in self.robot.active_joint_name_ordering
                assert active_jnt_name[1] in self.robot.active_joint_name_ordering

                # TODO: `direction` is weired...
                active_jnt_dir = {active_jnt_name[0]:1,
                           active_jnt_name[1]: _sysID_specs.direction}
                # joint_idx = [
                #     robot.active_joint_name_ordering.index(joint_names[0]),
                #     robot.active_joint_name_ordering.index(joint_names[1]),
                # ]

            logger.info(f'{_symm_jnt_name=:}, {active_jnt_name=:}')

            # --- calc warm up action:
            mean_angle = (
                robot.joint_cfg_limits[active_jnt_name[0]][0]
                + robot.joint_cfg_limits[active_jnt_name[0]][1]
            ) / 2.
            amplitude_max = robot.joint_cfg_limits[active_jnt_name[0]][1] - mean_angle
            logger.info(f'{mean_angle=:}, {amplitude_max=:}')

            # warm up sysID motor:
            # active_jnt_warm_up_angles[joint_idx[0]] = mean
            # active_jnt_warm_up_angle[jnt_name[0]] = mean
            # if len(jnt_name) > 1:
            #     active_jnt_warm_up_angle[jnt_name[1]] = mean * _sysID_specs.direction

            # warm up sysID motor:
            for _n, _d in active_jnt_dir:
                active_jnt_warm_up_angle[_n] = mean_angle * _d

            # warm up other accompany joints:
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
            #         ep_time_seq, ep_act_seq =self.build_episode(time_curr=self.sysID_time_seq[-1],
            #                            action_curr=self.sysID_motor_act_seq[-1],
            #                            sysID_jnt_direction=active_jnt_dir,
            #                            active_jnt_warm_up_angle=active_jnt_warm_up_angle,
            #                            duration_warm_up=_WARM_UP_DURATION,
            #                            duration_reset=_RESET_DURATION,
            #                            chirp_signal_param=chirp_param)
            #
            #         self.sysID_time_seq = np.concatenate([self.sysID_time_seq, ep_time_seq], axis=0, dtype=np.float32)
            #         self.sysID_motor_act_seq = np.concatenate([self.sysID_motor_act_seq, prep_motor_act_seq], axis=0, dtype=np.float32)
            #
            #         # use kp=0.0, no feedback?
            #         # TODO: episode_motor_kp indexed by episode start time index in `sysID_time_seq`.
            #         self.episode_motor_kp[self.sysID_time_seq[-1]] = dict(
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

            motor_name: List[str] = []
            for _n in active_jnt_name:
                motor_name.extend(self.robot.active_joint_to_motor_name[_n])

            for _kp, _ratio in product(kp_list, _sysID_specs.amplitude_ratio_list):
                chirp_param = dict(
                    duration=_SIGNAL_DURATION,
                    control_dt=self.control_dt,
                    mean=0.0,
                    initial_frequency=_sysID_specs.initial_frequency,
                    final_frequency=_sysID_specs.final_frequency,
                    amplitude=_ratio * amplitude_max,
                    decay_rate=_sysID_specs.decay_rate)

                ep_time_seq, ep_act_seq = self.build_episode(time_curr=self.sysID_time_seq[-1],
                                                             action_curr=self.sysID_motor_act_seq[-1],
                                                             sysID_jnt_direction=active_jnt_dir,
                                                             active_jnt_warm_up_angle=active_jnt_warm_up_angle,
                                                             duration_warm_up=_WARM_UP_DURATION,
                                                             duration_reset=_RESET_DURATION,
                                                             chirp_signal_param=chirp_param)

                self.sysID_time_seq = np.concatenate([self.sysID_time_seq, ep_time_seq], axis=0,
                                                     dtype=np.float32)
                self.sysID_motor_act_seq = np.concatenate([self.sysID_motor_act_seq, prep_motor_act_seq],
                                                          axis=0, dtype=np.float32)

                # motor_name: List[str] = []
                # for _n in active_jnt_name:
                #     motor_name.extend(self.robot.active_joint_to_motor_name[_n])

                # self.episode_motor_kp[self.sysID_time_seq[-1]] = { _n: _kp for _n in motor_name}
                # must be ordered list.
                self.episode_motor_kp.append(_EpisodeKp(ep_end_time_pnt=self.sysID_time_seq[-1],
                                                        motor_kp={ _n: _kp for _n in motor_name}))


                logger.info(f'--->build episode, end time: {self.sysID_time_seq[-1]}, end act: {self.sysID_motor_act_seq[-1]}'
                            f'\n active jnt name: {active_jnt_name}, active jnt direction: {active_jnt_dir}, '
                            f'\n active jnt warm_up angle: {active_jnt_warm_up_angle}, amplitude ratio: {_ratio}'
                            f'\n sysID_time_seq shape: {self.sysID_time_seq.shape}, sysID_act_seq shape: {self.sysID_motor_act_seq.shape} '
                            f'\n kp for motors: {self.episode_motor_kp[-1]} ')


                # self.episode_motor_kp[self.sysID_time_seq[-1]] = dict(
                #     zip(active_jnt_name, [kp] * len(active_jnt_name))
                # )

        # self.time_arr = np.concatenate(time_list)
        # self.action_arr = np.concatenate(action_list)

        # override the value set in BasePolicy.__init__()
        self.n_steps_total = len(self.sysID_time_seq)
        logger.info(f'finish building all the episodes: {self.n_steps_total=:}'
                    f'sysID end time: {self.sysID_time_seq[-1]}, end act: {self.sysID_motor_act_seq[-1]}')


    # one episode: act_seq from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up.
    def build_episode(self, *, time_curr: float,
                      action_curr: npt.NDArray[np.float32],
                      # action_warm_up: npt.NDArray[np.float32],
                      sysID_jnt_direction: Mapping[str, int],

                      # include warm_up sysID motors and other accompany active joints.
                      active_jnt_warm_up_angle: OrderedDict[str, float],

                      duration_warm_up: float,
                      duration_reset: float,
                      chirp_signal_param: Dict[str, float],
                      # kp: float
                      ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        assert action_curr.shape[0] == self.robot.nu

        # including from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up
        # shape: ( time_seq_len, robot.nu )
        episode_time_seq: npt.NDArray[np.float32] | None = None
        episode_act_seq: npt.NDArray[np.float32] | None = None

        # --- warm up:
        # TODO: len(active_jnt_warm_up_angles) != robot.nu......
        act_warm_up = np.zeros_like(self.robot.motor_name_ordering, dtype=np.float32)

        # `active_jnt_warm_up_angle` include warm_up sysID motors and other accompany active joints.
        for _n, _a in self.robot.active_joint_to_motor_angles(active_jnt_warm_up_angle):
            idx = self.robot.motor_name_ordering.index(_n)
            act_warm_up[idx] = _a

        if not np.allclose(act_warm_up, action_curr, 1e-06):  # self.action_arr[-1, :], 1e-6):
            warm_up_time_seq, warm_up_motor_act_seq = self.move(time_curr=time_curr,  # self.time[-1],
                                                                action_curr=action_curr,  # self.action_arr[-1, :],
                                                                action_next=act_warm_up,
                                                                duration=duration_warm_up)
            episode_time_seq = warm_up_time_seq
            episode_act_seq = warm_up_motor_act_seq

        # --- rotate joint angles by `chirp` signal:
        chirp_time_seq, chirp_signal_seq = get_chirp_signal(**chirp_signal_param)

        # if episode_time_seq is not None:
        #     chirp_time_seq += episode_time_seq[-1] + self.control_dt
        # else:
        #     chirp_time_seq += time_curr + self.control_dt

        # rotate_jnt_angle_seq = np.zeros((chirp_time_seq.shape[0], robot.nu), np.float32)
        # active_jnt_rotate_angle_seq = np.zeros((chirp_time_seq.shape[0], len(self.robot.active_joint_name_ordering), np.float32)
        # shape: ( time_seq_len, robot.nu )

        # only for sysID joint, not include accompany active joints.
        sysID_jnt_chirp_angle = OrderedDict( (_n, chirp_signal_seq * _d)
                                             for _n,_d in sysID_jnt_direction)

        # chirp_motor_act_seq = np.zeros(shape=(chirp_time_seq.shape[0], len(self.robot.motor_name_ordering)),
        #                                dtype=np.float32)

        # chirp_motor_act_seq = np.repeat(act_warm_up[np.newaxis,:], repeats=chirp_time_seq.shape[0], axis=0)
        chirp_motor_act_seq = np.tile(act_warm_up, reps=(chirp_time_seq.shape[0],1))  # shape: ( chirp_time_seq_len, robot.nu )

        for _n, _a in self.robot.active_joint_to_motor_angles(sysID_jnt_chirp_angle):
            idx = self.robot.motor_name_ordering.index(_n)
            # add chirp onto warm_up act.
            chirp_motor_act_seq[:, idx] += _a  # shape: len(chirp_time_seq)

        # add chirp onto warm_up act.
        # chirp_motor_act_seq[:] += act_warm_up  # shape: ( time_seq_len, robot.nu )

        # rotate_jnt_angle_seq[:, joint_idx[0]] = signal_seq
        # if len(joint_idx) > 1:
        #     rotate_jnt_angle_seq[:, joint_idx[1]] = signal_seq * sysID_specs.direction

        # rotate_act_seq = np.zeros_like(rotate_jnt_angle_seq)
        # for _t_idx, _jnt_angle in enumerate(rotate_motor_angle_seq):
        #     rotate_motor_angles = robot.active_joint_to_motor_angles(
        #         dict(zip(robot.active_joint_name_ordering, _jnt_angle))
        #     )
        #     signal_action = np.array(
        #         list(rotate_motor_angles.values()), dtype=np.float32
        #     )
        #     # add signal onto warm_up pos.
        #     rotate_act_seq[_t_idx] = signal_action + action_warm_up

        if episode_time_seq is not None:
            chirp_time_seq += episode_time_seq[-1] + self.control_dt
            episode_time_seq = np.concatenate([episode_time_seq, chirp_time_seq], axis=0, dtype=np.float32)
            episode_act_seq = np.concatenate([episode_act_seq, chirp_motor_act_seq], axis=0, dtype=np.float32)
        else:
            chirp_time_seq += time_curr + self.control_dt
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
            interpolate_action(obs.time, self.sysID_time_seq, self.sysID_motor_act_seq)
        )
        return {}, action


    # @property
    # def n_steps_total(self):
    #     return len(self.sysID_time_seq)