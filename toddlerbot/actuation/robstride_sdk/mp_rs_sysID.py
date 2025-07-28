from dataclasses import dataclass
from typing import (Dict, List, Optional, Tuple,
                    Mapping, OrderedDict)
from collections import OrderedDict
from itertools import product

import numpy as np
import numpy.typing as npt

if __name__ == '__main__':
    from pathlib import Path
    import time as timelib
    import logging

    from toddlerbot.sim import Obs
    from toddlerbot.utils import (get_chirp_signal, interpolate_action, config_logging, interpolate)
    from toddlerbot.policies import sysIDEpisodeInfo,RUN_POLICY_LOG_FOLDER_FMT
    from toddlerbot.policies._module_logger import logger
    from toddlerbot.visualization import *
    from toddlerbot.actuation.base_controller import JointState
    from toddlerbot.policies.implementations.sysID import get_ep_trajectory_stat
    from toddlerbot.actuation.robstride_control import RobStrideController

else:
    from ...sim import Obs
    from ...utils import ( get_chirp_signal, interpolate_action, interpolate )
    from .._module_logger import logger
    from ...policies import sysIDEpisodeInfo
    from ..base_controller import JointState
    from ..robstride_control import RobStrideController

# This script collects data for system identification of the motors.
# in seconds.
_WARM_UP_DURATION = 2.0
_CHIRP_SIGNAL_DURATION = 10.0
_CHIRP_START_FREQ = 0.1

# TODO: 3 is enough?
_CHIRP_END_FREQ = 6 # 10.
_CHIRP_DECAY_RATE = 0.1  #0.1
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
def _build_jnt_sysID_spec()->Mapping[str, _SysIDSpecs]:

    # NOTE: the key is joint name corresponding to `robot.active_joint` name.
    # Not the motor name, but must be 1-to-1 mapping to motor name.
    specs : Mapping[str, _SysIDSpecs] | None = None
    kp_list: List[float] =  [32,64,96]
    amplitude_ratio_list:List[float] = [0.5]

    # single motor joint.
    specs = {
        # "joint_0": _SysIDSpecs(amplitude_ratio_list=[0.25, 0.5, 0.75], kp_list=kp_list)
        "joint_0": _SysIDSpecs(amplitude_ratio_list=amplitude_ratio_list,
                               kp_list=kp_list)
    }

    return specs

class MpSysIDPolicy:

    def __init__(
        self, *,
            init_motor_pos: npt.NDArray[np.float32],
            jnt_cfg_limit: Tuple[float,float],
            control_dt_sec:float ):
        """Initializes the class with specified parameters and sets up system identification specifications for robot joints.

        Args:
            init_motor_pos (npt.NDArray[np.float32]): observed Initial motor positions as a NumPy array.
            control_dt_sec:
        Attributes:

        """

        self.control_dt_sec = control_dt_sec

        # some default values. can be overridden.
        # self.control_dt_sec: float = 0.02  # 20ms, 50Hz.
        self.prep_duration_sec: float = 2.0
        # self.n_steps_total: float = float("inf")

        jnt_sysID_specs = _build_jnt_sysID_spec()

        self.episode_info: List[sysIDEpisodeInfo] = []

        # NOTE: `act` is motor control action, e.g., target pos of motor.
        # In prep duration, make all motors back to zero.
        # In the following steps, the default angels for irrelative motors are kept `0`. easy way.
        prep_time_seq, prep_motor_act_seq = self.interpolate_traj(time_curr= -self.control_dt_sec,
                                                                  action_curr=init_motor_pos,
                                                                  action_next= np.zeros_like(init_motor_pos),
                                                                  duration=self.prep_duration_sec)

        # for whole sysID procedure.
        self._sysID_time_seq: npt.NDArray[np.float32] = prep_time_seq
        self._sysID_motor_act_seq: npt.NDArray[np.float32] = prep_motor_act_seq
        self.n_steps_total: float = -1.

        logger.info(f'{init_motor_pos=:}')

        # NOTE: guarantee only one sysID_joint per one episode. but one sysID_joint has multiple
        # episode with different kp/ampl.
        for _symm_jnt_name, _sysID_specs in jnt_sysID_specs.items():
            # joint_idx: List[int] | None = None
            sysID_jnt_name : List[str] | None = None
            sysID_jnt_dir: Mapping[str, int] | None = None

            sysID_jnt_name = [_symm_jnt_name]
            # joint_idx = [robot.active_joint_name_ordering.index(joint_names[0])]
            sysID_jnt_dir = {sysID_jnt_name[0]:1}

            logger.info(f'{_symm_jnt_name=:}, {sysID_jnt_name=:}, {sysID_jnt_dir=:}')

            # --- calc warm up action:
            mean_angle = (
                jnt_cfg_limit[0]
                + jnt_cfg_limit[1]
            ) / 2.
            amplitude_max = jnt_cfg_limit[1] - mean_angle
            logger.info(f'{mean_angle=:}, {amplitude_max=:}')

            # NOTE: assign valid values for warm_up sysID joints and other accompany active joints.
            active_jnt_warm_up_angle: OrderedDict[str, float] = OrderedDict(
                [
                    ('joint_0',0.),
                 ]
            )

            # warm up sysID motor:
            for _n, _d in sysID_jnt_dir.items():
                active_jnt_warm_up_angle[_n] = mean_angle * _d

            if _sysID_specs.kp_list is None:
                # TODO: why use 0, PD controller not work? how about use defaults?
                kp_list = [0.]
            else:
                kp_list =_sysID_specs.kp_list

            sysID_motor_name: List[str] = sysID_jnt_name

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


        # override the value set in BasePolicy.__init__()
        self.n_steps_total = len(self._sysID_time_seq)
        logger.info(f'finish building all the episodes: {self.n_steps_total=:} '
                    f'sysID end time: {self._sysID_time_seq[-1]}, end act: {self._sysID_motor_act_seq[-1]}')


    # one episode: act_seq from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up.
    # NOTE: in one episode, can have 1~2 sysID_joints.
    def _build_motor_act_episode(self, *, time_curr: float,
                                action_curr: npt.NDArray[np.float32],

                                # NOTE: only for sysID active jnt, not include the accompany
                                # active joints for warm_up. we add `chirp` signal only onto sysID joints.
                                sysID_jnt_direction: Mapping[str, int],

                                # NOTE: include warm_up sysID motors and other accompany active joints.
                                # so not to add chirp signal to all joints among `active_jnt_warm_up_angle`.
                                active_jnt_warm_up_angle: OrderedDict[str, float],

                                duration_warm_up: float,
                                duration_reset: float,
                                chirp_signal_param: Dict[str, float],

                                ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        # at most 2 sysID joint per episode.
        assert len(sysID_jnt_direction) <= 2

        # including from action_curr -> warm_up -> chirp_signal -> reset_to_warm_up
        # shape: ( time_seq_len, robot.nu )
        episode_time_seq: npt.NDArray[np.float32] | None = None
        episode_act_seq: npt.NDArray[np.float32] | None = None


        # `active_jnt_warm_up_angle` include warm_up sysID motors and other accompany active joints.
        motor_warm_up_angle = active_jnt_warm_up_angle

        act_warm_up: npt.NDArray[np.float32] = np.asarray([motor_warm_up_angle['joint_0']],
                                                          dtype=np.float32)

        if not np.allclose(act_warm_up, action_curr, 1e-06):  # self.action_arr[-1, :], 1e-6):
            warm_up_time_seq, warm_up_motor_act_seq = self.interpolate_traj(time_curr=time_curr,  # self.time[-1],
                                                                            action_curr=action_curr,  # self.action_arr[-1, :],
                                                                            action_next=act_warm_up,
                                                                            duration=duration_warm_up)
            episode_time_seq = warm_up_time_seq
            episode_act_seq = warm_up_motor_act_seq

        # --- rotate joint angles by `chirp` signal.
        # NOTE: only add chirp signal onto sysID joints, not add onto `active_jnt_warm_up_angle`...
        chirp_signal_seq:npt.NDArray[np.float32]
        chirp_time_seq, chirp_signal_seq = get_chirp_signal(**chirp_signal_param)


        # construct joint angle first:   {active_jnt_name: jnt_angle_seq ... }
        active_jnt_chirp_angle_seq = OrderedDict(
            (_n, chirp_signal_seq * sysID_jnt_direction[_n] if _n in sysID_jnt_direction
                else np.zeros_like(chirp_signal_seq,dtype=np.float32) )
            for _n in ['joint_0'] )

        motor_chirp_angle_seq: OrderedDict[str, npt.NDArray[np.float32]] =  active_jnt_chirp_angle_seq

        # shape: (robot.nu, len(chirp_time_seq) ) -> shape: (len(chirp_time_seq), robot.nu)
        chirp_motor_act_seq: npt.NDArray[np.float32] = np.asarray(
            [motor_chirp_angle_seq[_n] for _n in ['joint_0'] ],
            dtype=np.float32).transpose()

        logger.info(f' {chirp_motor_act_seq.shape=:} ')

        # add chirp onto warm_up act.
        chirp_motor_act_seq[:] += act_warm_up  # shape: ( len(chirp_time_seq), robot.nu )

        if episode_time_seq is not None:
            chirp_time_seq += episode_time_seq[-1] + self.control_dt_sec
            episode_time_seq = np.concatenate([episode_time_seq, chirp_time_seq], axis=0, dtype=np.float32)
            episode_act_seq = np.concatenate([episode_act_seq, chirp_motor_act_seq], axis=0, dtype=np.float32)
        else:
            chirp_time_seq += time_curr + self.control_dt_sec
            episode_time_seq = chirp_time_seq
            episode_act_seq = chirp_motor_act_seq

        # --- reset to warm up:
        reset_time_seq, motor_reset_act_seq = self.interpolate_traj(time_curr=episode_time_seq[-1],
                                                                    action_curr=episode_act_seq[-1, :],
                                                                    action_next=act_warm_up,
                                                                    duration=duration_reset,
                                                                    end_time=0.5)

        # NOTE: already reset_time_seq += episode_time_seq[-1] + self.control_dt in self.move().
        episode_time_seq = np.concatenate([episode_time_seq, reset_time_seq], axis=0, dtype=np.float32)
        episode_act_seq = np.concatenate([episode_act_seq, motor_reset_act_seq], axis=0, dtype=np.float32)

        return episode_time_seq, episode_act_seq

    def interpolate_traj(
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
            num=int(duration / self.control_dt_sec),
            endpoint=False,
            dtype=np.float32,
        )

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
        time_seq += time_curr + self.control_dt_sec

        return time_seq, act_seq


    def step(
        self, jnt_state: JointState,
            is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the environment by interpolating an action based on the given observation time.

        Args:
            jnt_state (Obs): The observation containing the current time.
            is_real (bool, optional): Flag indicating whether the step is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the interpolated action as a NumPy array.
        """

        action = np.asarray(
            interpolate_action(jnt_state.time, self._sysID_time_seq, self._sysID_motor_act_seq),
            dtype=np.float32
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

    def pre_process(self):
        pass

    def post_process(self):
        pass


# def get_ep_trajectory_stat(policy:MpSysIDPolicy)\
#         ->Dict[str,OrderedDict[str,npt.NDArray[np.float32]]]:
#
#     time_diff = np.diff(policy._sysID_time_seq, n=1, axis=0)
#     # calc vel
#     act_diff = np.diff(policy._sysID_motor_act_seq, n=1, axis=0)
#
#     # rad/s
#     target_vel = (act_diff.transpose() / time_diff).transpose()
#     # trick: keep dimension consistent as len(time_seq).
#     target_vel = np.concatenate([np.zeros_like(target_vel[0], dtype=np.float32)[np.newaxis, :], target_vel],
#                               axis=0, dtype=np.float32)
#
#     # rpm
#     target_rpm = target_vel * 60/(2*np.pi)
#
#     # calc acc
#     vel_diff = np.diff(target_vel, n=1, axis=0)
#     # rad/s**2
#     # target_acc = (vel_diff.transpose() / time_diff[1:]).transpose()
#     target_acc = (vel_diff.transpose() / time_diff).transpose()
#     # trick: keep dimension consistent as len(time_seq).
#     target_acc = np.concatenate([np.zeros_like(target_acc[0], dtype=np.float32)[np.newaxis, :], target_acc],
#                                 axis=0, dtype=np.float32)
#
#     # round per s**2
#     target_acc_rps2 = target_acc/(2*np.pi)
#
#     logger.info(f'--- episode trajectory vel / acc stats: ---')
#     logger.info(f'chirp_max_freq:{_CHIRP_END_FREQ}Hz {act_diff.shape=:} {time_diff.shape=:} '
#                 f'{target_vel.shape=:} {target_rpm.shape=:} {target_acc.shape=:}')
#
#     logger.warning(f'{max(act_diff)=:} {max(time_diff)=:} {max(target_vel)=:} rad/s '
#                    f'\n{max(target_rpm)=:} {max(target_acc)=:} rad/s**2 {max(target_acc_rps2)=:} round/s**2')
#
#     # construct dict:
#
#     target_pos_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#     pos_time_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#
#     target_acc_rps2_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#     acc_time_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#
#     target_rpm_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#     vel_time_dict: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
#
#     for _x, _n in enumerate(['joint_0']):
#         target_pos_dict[_n] = policy._sysID_motor_act_seq[:, _x]
#         pos_time_dict[_n] = policy._sysID_time_seq
#
#         target_acc_rps2_dict[_n] = target_acc_rps2[:, _x]
#         acc_time_dict[_n] = policy._sysID_time_seq #[2:]
#
#         target_rpm_dict[_n] = target_rpm[:, _x]
#         vel_time_dict[_n] = policy._sysID_time_seq #[1:]
#
#     return   { 'target_pos_dict': target_pos_dict,
#                'pos_time_dict': pos_time_dict,
#                'target_acc_rps2_dict': target_acc_rps2_dict,
#                'acc_time_dict': acc_time_dict,
#                'target_rpm_dict': target_rpm_dict,
#                'vel_time_dict': vel_time_dict,
#                }


@dataclass(init=True)
class MotorKpSetter:
    _cur_ep_idx: int = -1

    def set_kp(self, *,
               policy: MpSysIDPolicy,
               ctrl: RobStrideController,
               step_count: int,
               obs_time: float):
        # assert isinstance(policy, SysIDPolicy)
        assert type(policy).__name__ == 'MpSysIDPolicy'
        # always set first ep kp.
        if step_count == 0 or self._cur_ep_idx == -1:
            # we do not allow obs skip an episode.
            assert obs_time <= policy.episode_info[0].ep_end_time_pnt
            self._cur_ep_idx = 0

            ctrl.set_pos_kp(list(policy.episode_info[self._cur_ep_idx].motor_kp.values()))

            logger.info(f'update cur episode idx to {self._cur_ep_idx}, '
                        f'and set motor kp: {policy.episode_info[self._cur_ep_idx].motor_kp}')

        # TODO: if len(ep) == 1?
        elif self._cur_ep_idx == len(policy.episode_info) - 1:
            # already last episode, not set.
            pass

        elif obs_time > policy.episode_info[self._cur_ep_idx].ep_end_time_pnt:
            # we do not allow obs skip an episode.
            # assert obs_time <= policy.episode_info[self._cur_ep_idx + 1].ep_end_time_pnt
            if obs_time > policy.episode_info[self._cur_ep_idx + 1].ep_end_time_pnt:
                raise ValueError(f'we do not allow obs skip an episode: '
                                 f'{obs_time=:} > {policy.episode_info[self._cur_ep_idx + 1].ep_end_time_pnt =:}')

            self._cur_ep_idx += 1

            ctrl.set_pos_kp(list(policy.episode_info[self._cur_ep_idx].motor_kp.values()))

            logger.info(f'update cur episode idx to {self._cur_ep_idx}, '
                        f'and set motor kp: {policy.episode_info[self._cur_ep_idx].motor_kp}')

        else:
            # kp no change, not set.
            pass


class MockRobot:
    def __init__(self, name: str):
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def motor_name_ordering(self) -> List[str]:
        return ['joint_0', ]

    @property
    def joint_cfg_limits(self)->Tuple[float,float]:
        return -np.pi/2, np.pi/2

    def motor_to_active_joint_angles(self,  # joints_config: Mapping[str, Any],
                                     motor_angles: OrderedDict[str, float | npt.NDArray[np.float32]],
                                     ) -> OrderedDict[str, float | npt.NDArray[np.float32]]:
        return motor_angles


def _test_main():

    config_logging(root_logger_level=logging.INFO, root_handler_level=logging.NOTSET,
                   # root_fmt='--- {levelname} - module:{module} - func:{funcName} ---> \n{message}',
                   root_fmt='--- {levelname} - module:{module} ---> \n{message}',
                   root_date_fmt='%Y-%m-%d %H:%M:%S',
                   # log_file='/tmp/toddler/imitate_episode.log',
                   log_file=None,
                   module_logger_config={'mp_rs_sysID':logging.INFO,
                                         })

    # use root logger for __main__.
    # logger = logging.getLogger('root')

    # like normalized value.
    init_motor_pos = np.zeros_like(['joint_0'], dtype=np.float32)
    policy = MpSysIDPolicy(init_motor_pos=init_motor_pos,
                           jnt_cfg_limit=(-np.pi/2, np.pi/2),
                           control_dt_sec=0.04)

    mock_rbt = MockRobot('test_mp_sysID')

    stat_dict = get_ep_trajectory_stat(robot=mock_rbt,
                                       policy=policy)

    exp_folder = Path(RUN_POLICY_LOG_FOLDER_FMT.format(robot_name='sysID_RS02',
                                                       policy_name='MpRSSysID',
                                                       env_name='test_episode',
                                                       cur_time=timelib.strftime("%Y%m%d_%H%M%S")))
    plot_dir = exp_folder / 'sysID_episode_trajectory_plot'
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    plot_joint_tracking_single(
        time_seq_dict=stat_dict['pos_time_dict'],
        joint_data_dict=stat_dict['target_pos_dict'],
        save_path=plot_dir.resolve().__str__(),
        x_label="Time (s)",
        y_label="Pos (rad)",
        file_name="target_pos_trajectory",
        set_ylim=False,
    )

    plot_joint_tracking_single(
        time_seq_dict=stat_dict['vel_time_dict'],
        joint_data_dict=stat_dict['target_rpm_dict'],
        save_path=plot_dir.resolve().__str__(),
        x_label="Time (s)",
        y_label="Vel (RPM)",
        file_name = "target_vel_rpm_trajectory",
        set_ylim=False,
    )

    plot_joint_tracking_single(
        time_seq_dict=stat_dict['acc_time_dict'],
        joint_data_dict=stat_dict['target_acc_rps2_dict'],
        save_path=plot_dir.resolve().__str__(),
        x_label="Time (s)",
        y_label="Acc (RP/S**2)",
        file_name="target_acc_rps2_trajectory",
        set_ylim=False,
    )

if __name__ == '__main__':
    _test_main()
