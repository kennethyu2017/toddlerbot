import argparse
import json
from pathlib import Path
import logging
import pickle
import time
from typing import Dict, List, Tuple, Generator, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass,field

import numpy as np
import numpy.typing as npt
import optuna
# from optuna.logging import _get_library_root_logger

from toddlerbot.sim import Robot, MuJoCoSim, MotorController
from toddlerbot.utils import config_logging
from toddlerbot.policies import (RUN_POLICY_LOG_FOLDER_FMT, RUN_STEP_RECORD_PICKLE_FILE,
                                 RUN_EPISODE_MOTOR_KP_PICKLE_FILE, StepRecord, sysIDEpisodeInfo)

# from toddlerbot.utils.misc_utils import log

from toddlerbot.visualization import (
    plot_joint_tracking,
    plot_joint_tracking_frequency,
)

from toddlerbot.tools._module_logger import logger

# This script is used to optimize the parameters of the robot's dynamics model using system identification (SysID) techniques.

# TODO> use optuna logger?
# logger = _get_library_root_logger()

# def _set_obs_and_action_helper(
#         joint_name: str, motor_kps: Dict[str, float], idx_range: slice
# ):
#     # kp = motor_kps.get(joint_name, 0)
#
#     obs_pos_list: List[List[float]] = []
#     for obs in obs_list[idx_range]:
#         motor_angles_obs = dict(zip(robot.motor_name_ordering, obs.motor_pos))
#         joint_angles_obs = robot.motor_to_active_joint_angles(motor_angles_obs)
#         obs_pos_list.append(list(joint_angles_obs.values()))
#
#     obs_pos = np.array(obs_pos_list)
#
#     action = np.array(
#         [
              # `motor_angles_list` is a list of `motor_target_arr`.i.e., action.
#             list(motor_angles.values())
#             for motor_angles in motor_angles_list[idx_range]
#         ]
#     )
#
#     if joint_name not in obs_pos_dict:
#         obs_pos_dict[joint_name] = []
#         action_dict[joint_name] = []
#         kp_dict[joint_name] = []
#
#     NOTE: (obs_pos, action, kp) belong to one episode of one sysID_jnt, and one sysID_jnt has
#                                 multiple episodes.
#     obs_pos_dict[joint_name].append(obs_pos)
#     action_dict[joint_name].append(action)
#     kp_dict[joint_name].append(kp)
#

@dataclass(init=True)
class _SysIDEpisodeData:
    # for a single sysID joint run log data. each episode, has 1 or 2 sysID joints.
    sysID_jnt_name: str  = field(default_factory=str)
    sysID_jnt_pos: List[float] = field(default_factory=list)            # shape: ( len of episode_time_seq, )
    motor_act: List[npt.NDArray[np.float32]] = field(default_factory=list)    # shape: ( len of episode_time_seq, robot.nu)
    motor_kp: Dict[str, float] = field(default_factory=dict)                     #  {motor_name:kp...}


# yield one episode per iter.
def _load_dataset(*,robot: Robot, step_record_file: Path, kp_file: Path)->Generator[_SysIDEpisodeData, None, None]:
    #  ->Tuple[Dict[str, List[npt.NDArray[np.float32]]],Dict[str, List[npt.NDArray[np.float32]]],Dict[str, List[float]]]:
    """Loads and processes datasets from a specified path for a given robot, extracting observation positions, actions, and motor gains.

    Args:
        robot (Robot): The robot instance containing motor and joint configurations.
        step_record_file (str): The directory path where the dataset files are located.
        kp_file

    Returns:
        Tuple[Dict[str, List[npt.NDArray[np.float32]]], Dict[str, List[npt.NDArray[np.float32]]], Dict[str, List[float]]]:
        A tuple containing three dictionaries:
            - obs_pos_dict: Maps joint names to lists of observation position arrays.
            - action_dict: Maps joint names to lists of action arrays.
            - kp_dict: Maps joint names to lists of motor gain values.

    Raises:
        ValueError: If no data files are found at the specified path.
    """

    # Use glob to find all pickle files matching the pattern
    # pickle_file: Path = data_path /"log_data.pkl"
    # if not pickle_file.exists():
    #     raise ValueError(f"No pickle data file found: {pickle_file.resolve()} ")

    step_record_list: List[StepRecord] |None = None
    with open(step_record_file, "rb") as _f:
        step_record_list = pickle.load(_f)

    # obs_list: List[Obs] = data_dict["obs_list"]
    # motor_name: target_angle
    # motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]

    # obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    # action_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    # kp_dict: Dict[str, List[float]] = {}

    # def set_obs_and_action(
    #     joint_name: str, motor_kps: Dict[str, float], idx_range: slice
    # ):
    #     kp = motor_kps.get(joint_name, 0)
    #
    #     obs_pos_list: List[List[float]] = []
    #     for obs in obs_list[idx_range]:
    #         motor_angles_obs = dict(zip(robot.motor_name_ordering, obs.motor_pos))
    #         joint_angles_obs = robot.motor_to_active_joint_angles(motor_angles_obs)
    #         obs_pos_list.append(list(joint_angles_obs.values()))
    #
    #     obs_pos = np.array(obs_pos_list)
    #
    #     action = np.array(
    #         [
    #             list(motor_angles.values())
    #             for motor_angles in motor_angles_list[idx_range]
    #         ]
    #     )
    #
    #     if joint_name not in obs_pos_dict:
    #         obs_pos_dict[joint_name] = []
    #         action_dict[joint_name] = []
    #         kp_dict[joint_name] = []
    #
    #     obs_pos_dict[joint_name].append(obs_pos)
    #     action_dict[joint_name].append(action)
    #     kp_dict[joint_name].append(kp)

    # 'joint0':
    # sysID_jnt_obs_pos_list: List[ npt.NDArray[np.float32]] = []
    # motor_act_list: [ List[ npt.NDArray[np.float32]] ] = []
    # kp_list:  List[float] = []

    # if "ckpt_dict" in data_dict:
    # TODO> why not exist?
    if kp_file.exists():
        # ckpt_dict: Dict[str, Dict[str, float]] = data_dict["ckpt_dict"]
        # list of {motor_name:kp ... } of each episode.
        # each episode, only one sysID_joint, but maybe more than one corresponding motors.
        episode_info: List[sysIDEpisodeInfo] | None = None
        with open(kp_file,'rb') as _f:
            episode_info = pickle.load(_f)

        sysID_ep_data: List[_SysIDEpisodeData] = []
        _cur_ep_idx: int = -1
        _new_episode: bool=False

        # TODO: how to yield at last record?
        # TODO: start record index = 200 ?
        for _r in step_record_list:
            if _cur_ep_idx == -1:
                # we do not allow obs skip an episode.
                assert _r.obs.time <= episode_info[0].ep_end_time_pnt
                _cur_ep_idx = 0
                logger.info(f'update cur episode idx to {_cur_ep_idx})')
                _new_episode = True

            #
            # elif _cur_ep_idx == len(episode_info) - 1:
            #     # already last episode.
            #     _new_episode=False
            #     pass

            elif _r.obs.time > episode_info[_cur_ep_idx].ep_end_time_pnt:
                if _cur_ep_idx == len(episode_info) - 1:
                    # already last episode. not yield here causing there can still have
                    # obs coming after. we yield after exiting from loop.
                    _new_episode = False

                else:
                    # to next episode.
                    logger.info(f'finish current sysID episode: {_cur_ep_idx} '
                                f'cur episode end time: {episode_info[_cur_ep_idx].ep_end_time_pnt}')

                    # TODO: deepcopy?
                    for _d in sysID_ep_data:
                        logger.info(f'yield sysID episode record data : sysID jnt name: {_d.sysID_jnt_name=:} '
                                    f'motor kp dict: {_d.motor_kp} '
                                    f'len of jnt pos: {len(_d.sysID_jnt_pos)} '
                                    f'shape of motor action: {(len(_d.motor_act), _d.motor_act[0].shape)}')
                        yield _d

                    # we do not allow obs skip an episode.
                    assert _r.obs.time <= episode_info[_cur_ep_idx + 1].ep_end_time_pnt
                    _cur_ep_idx += 1
                    _new_episode = True

            else:
                # no change.
                _new_episode=False
                pass

            if _new_episode:
                logger.info(f'start new episode: {_cur_ep_idx}, '
                            f'end time: {episode_info[_cur_ep_idx].ep_end_time_pnt}'
                            f'sysID jnt name" {episode_info[_cur_ep_idx].sysID_jnt_name}')
                # each episode has 1 or 2 sysID joints.
                sysID_ep_data = [_SysIDEpisodeData(sysID_jnt_name=_n, motor_kp=episode_info[_cur_ep_idx].motor_kp)
                                      for _n in episode_info[_cur_ep_idx].sysID_jnt_name]

                # for _d in sysID_jnt_opt_data:
                #     logger.info(f'new episode sysID jnt name: {_d.sysID_jnt_name=:} ')

                _new_episode = False


            # NOTE: real world only collect motor_pos/motor_vel in Obs, can not use
            # joint_pos/joint_vel in Obs from real world.
            motor_pos_dict = OrderedDict(
                # (_n, _r.obs.motor_pos[_i]) for _i,_n in enumerate(robot.motor_name_ordering)
                zip(robot.motor_name_ordering, _r.obs.motor_pos)
            )
            active_joint_angle: OrderedDict[str, float] = robot.motor_to_active_joint_angles(motor_pos_dict)

            for _d in sysID_ep_data:
                assert _r.motor_act.shape == (robot.nu,)
                _d.motor_act.append(_r.motor_act)

                # # NOTE: real world only collect motor_pos/motor_vel in Obs, can not use
                # # joint_pos/joint_vel in Obs from real world.
                # motor_pos_dict = OrderedDict(
                #     # (_n, _r.obs.motor_pos[_i]) for _i,_n in enumerate(robot.motor_name_ordering)
                #     zip(robot.motor_name_ordering, _r.obs.motor_pos)
                # )
                #
                # active_joint_angle:OrderedDict[str, float] = robot.motor_to_active_joint_angles(motor_pos_dict)
                # jnt_ordering_idx = robot.active_joint_name_ordering.index(_d.sysID_jnt_name)

                _d.sysID_jnt_pos.append(active_joint_angle[_d.sysID_jnt_name] )

        #finish loop, always yield last opt data...
        assert len(sysID_ep_data) > 0
        for _d in sysID_ep_data:
            logger.info(f'yield LAST sysID episode record data : sysID jnt name: {_d.sysID_jnt_name=:} '
                        f'motor kp dict: {_d.motor_kp} '
                        f'len of jnt pos: {len(_d.sysID_jnt_pos)} '
                        f'shape of motor action: {(len(_d.motor_act), _d.motor_act[0].shape)}')
            yield _d


        # ckpt_times = list(ckpt_dict.keys())
        # ep_end_time_list = ( _e.ep_end_time_pnt for _e in episode_motor_kp )

        # motor_kps_list: List[Dict[str, float]] = []
        # joint_names_list: List[List[str]] = []
        # for d in list(ckpt_dict.values()): # each value is an episode.
        #     motor_kps_list.append(d)
        #     TODO: original code is sysID joint name, and kenneth changed them to motor name...
            # joint_names_list.append(list(d.keys()))

        # obs_time = (obs.time for obs in obs_list)
        # obs_indices = np.searchsorted(obs_time, ckpt_times)  #return indices : array of insertion points with the same shape as `ckpt_times`,
        # or an integer if `v` is a scalar.

        # TODO: 200 * 0.02s = 4sec. causing each episode > 10 sec, we can start from 4sec.
        # last_idx = 200
        # for joint_names, motor_kps, obs_idx in zip(
        #     joint_names_list, motor_kps_list, obs_indices
        # ):
        #     for joint_name in joint_names:
        #         # if "ank_roll" in joint_name:
        #         #     break
        #         # obs_idx's corresponding time is the end of each episode...
        #         _set_obs_and_action_helper(joint_name, motor_kps, slice(last_idx, obs_idx))
        #
        #     last_idx = obs_idx

    else:
        start_idx = 300
        for _jnt_name in reversed(robot.active_joint_name_ordering):
            joints_config = robot.config["joints"]
            if joints_config[_jnt_name]["group"] == "leg":
                motor_name:List[str] = robot.active_joint_to_motor_name[_jnt_name]
                # motor_kps = {_jnt_name: joints_config[motor_names[0]]["kp_real"]}
                motor_kp = {_n: joints_config[_n]["kp_real"] for _n in motor_name}
                # set_obs_and_action(joint_name, motor_kps, slice(start_idx, None))

                opt_data =  _SysIDEpisodeData(sysID_jnt_name=_jnt_name,
                                              motor_kp=motor_kp)

                # TODO> why not separate record into episodes?
                for _r in step_record_list[start_idx:]:
                    assert _r.motor_act.shape == (robot.nu,)
                    opt_data.motor_act.append(_r.motor_act)

                    # NOTE: real world only collect motor_pos/motor_vel in Obs, can not use
                    # joint_pos/joint_vel in Obs from real world.
                    motor_pos_dict = OrderedDict(
                        # (_n, _r.obs.motor_pos[_i]) for _i,_n in enumerate(robot.motor_name_ordering)
                        zip(robot.motor_name_ordering, _r.obs.motor_pos)
                    )

                    active_joint_angle: OrderedDict[str, float] = robot.motor_to_active_joint_angles(motor_pos_dict)
                    opt_data.sysID_jnt_pos.append(active_joint_angle[_jnt_name])

                yield opt_data

    # return obs_pos_dict, action_dict, kp_dict

def _build_early_stop(early_stop_rounds: int)\
        ->Callable[[optuna.Study, optuna.Trial], None]:

    def early_stop_check(
        study: optuna.Study, trial: optuna.Trial,
    ):
        """Checks if the current trial should trigger early stopping based on the number of rounds since the best trial.

        Args:
            study (optuna.Study): The study object containing all trials.
            trial (optuna.Trial): The current trial being evaluated.
            # early_stopping_rounds (int): The number of rounds to wait before stopping after the best trial.

        Logs a debug message and stops the study if early stopping conditions are met.
        """
        current_trial_number = trial.number
        best_trial_number = study.best_trial.number
        should_stop = (
            current_trial_number - best_trial_number
        ) >= early_stop_rounds
        if should_stop:
            logger.debug(f"early stopping detected: {should_stop}")
            study.stop()

    return early_stop_check



def _sim_run_sysID_episode_helper(*, ep_list:List[_SysIDEpisodeData],
                                  jnt_name:str,
                                  jnt_ordering_idx:int,
                                  sim:MuJoCoSim,
                                  )->npt.NDArray[np.float32]:
    jnt_pos_sim_list: List[float] = []
    # each `action` is a sequence belongs to one episode with a kp value.
    # for action, kp in zip(action_list, kp):
    for _ep in ep_list:
        # TODO: why not adjust joint pos in Mujoco at beginning of each episode ?
        # sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))
        sim.set_motor_kps(_ep.motor_kp )
        logger.info(f'set jnt: {jnt_name} kp: {_ep.motor_kp}')

        # for a in action:
        for _act in _ep.motor_act:
            obs = sim.get_observation(1)
            sim.set_motor_target(_act)
            sim.step()

            assert obs.joint_pos is not None
            jnt_pos_sim_list.append(obs.joint_pos[jnt_ordering_idx])

    jnt_pos_sim_arr = np.asarray(jnt_pos_sim_list,dtype=np.float32)
    logger.info(f'jnt: {jnt_name} joint_pos_sim_arr shape:{jnt_pos_sim_arr.shape}')

    return jnt_pos_sim_arr


def _build_objective(*,
                     robot:Robot,
                     sim: MuJoCoSim | None,
                     jnt_name: str,
                     ep_list: List[_SysIDEpisodeData],
                     freq_max: float = 10,
                     # gain_range: Tuple[float, float, float] ,
                     damping_range: Tuple[float, float, float],
                     armature_range: Tuple[float, float, float],
                     frictionloss_range: Tuple[float, float, float],
                     q_dot_tau_max_range: Tuple[float, float, float],
                     q_dot_max_range: Tuple[float, float, float],
                     tau_max_range: Tuple[float,float,float])\
        ->Callable[[optuna.Trial], float]:

    jnt_ordering_idx = robot.active_joint_name_ordering.index(jnt_name)
    robot_name = robot.name
    jnt_pos_real_arr: npt.NDArray[np.float32] = np.concatenate([_ep.sysID_jnt_pos for _ep in ep_list])
    logger.info(f'joint_pos_real_arr for jnt: {jnt_name}  shape:{jnt_pos_real_arr.shape}')

    def objective(trial: optuna.Trial)->float:
        """Optimize simulation parameters to minimize the error between simulated and real joint positions.

        This function uses Optuna to suggest values for various simulation parameters, including damping, armature, and friction loss, to optimize the joint dynamics of a robot simulation. If the robot's name contains "sysID", additional motor dynamics parameters are also optimized. The function calculates the root mean square error (RMSE) between the simulated and real joint positions and performs a Fourier Transform to compare the frequency domain characteristics, returning a combined error metric.

        Args:
            trial (optuna.Trial): An Optuna trial object used to suggest parameter values.

        Returns:
            float: The combined error metric, consisting of the RMSE and a weighted frequency domain error.
        """
        # gain = trial.suggest_float("gain", *gain_range[:2], step=gain_range[2])

        # TODO: each run of objective, will rollout all the episodes again, so we should reset sim to begging....
        # sim.reset....
        # TODO : use individual sim env for each objective, parallel...
        sim = MuJoCoSim(robot, fixed_base=True)

        damping = trial.suggest_float(
            "damping", *damping_range[:2], step=damping_range[2]
        )
        armature = trial.suggest_float(
            "armature", *armature_range[:2], step=armature_range[2]
        )
        frictionloss = trial.suggest_float(
            "frictionloss", *frictionloss_range[:2], step=frictionloss_range[2]
        )
        joint_dyn = {
            jnt_name: dict(
                damping=damping, armature=armature, frictionloss=frictionloss
            )
        }
        sim.set_joint_dynamics(joint_dyn)

        if "sysID" in robot_name:
            tau_max = trial.suggest_float(
                "tau_max", *tau_max_range[:2], step=tau_max_range[2]
            )
            q_dot_tau_max = trial.suggest_float(
                "q_dot_tau_max", *q_dot_tau_max_range[:2], step=q_dot_tau_max_range[2]
            )
            q_dot_max = trial.suggest_float(
                "q_dot_max", *q_dot_max_range[:2], step=q_dot_max_range[2]
            )

            # TODO: tau_max, q_dot_max, q_dot_tqu_max is array in mujoco_controller.....
            # sim.set_motor_dynamics(
            #     dict(tau_max=tau_max, q_dot_tau_max=q_dot_tau_max, q_dot_max=q_dot_max)
            # )
            # TODO: for SysID, wo only allow one motor now.
            motor_name: List[str] = robot.active_joint_to_motor_name[jnt_name]
            # this joint has more than one corresponding motors, e.g. , waist_roll  -> waist_act_1, waist_act_2.
            if not len(motor_name) == 1:
                raise ValueError(f'TBD: we do not support more than one motor name for sysID.'
                                 f'corresponding motor name: {motor_name}')

            sim.set_motor_dynamics(
                { motor_name[0] : dict(
                    tau_max= tau_max,
                    q_dot_tau_max= q_dot_tau_max,
                    q_dot_max= q_dot_max, )
                }
            )

        # joint_pos_sim: List[npt.NDArray[np.float32]] = []
        jnt_pos_sim_arr:npt.NDArray[np.float32] = _sim_run_sysID_episode_helper(ep_list=ep_list,
                                      jnt_name=jnt_name,
                                      jnt_ordering_idx=jnt_ordering_idx,
                                      sim=sim,
                                      )
        assert jnt_pos_real_arr.shape == jnt_pos_sim_arr.shape

        # RMSE
        error = np.sqrt(np.mean( (jnt_pos_real_arr - jnt_pos_sim_arr) ** 2) )

        # FFT (Fourier Transform) of the joint position data and reference data
        joint_pos_sim_fft = np.fft.fft(jnt_pos_sim_arr)
        joint_pos_real_fft = np.fft.fft(jnt_pos_real_arr)

        joint_pos_sim_fft_freq = np.fft.fftfreq(len(joint_pos_sim_fft), d=sim.dt)
        joint_pos_real_fft_freq = np.fft.fftfreq(len(joint_pos_real_fft), d=sim.dt)

        magnitude_sim = np.abs(joint_pos_sim_fft[: len(joint_pos_sim_fft) // 2])
        magnitude_real = np.abs(joint_pos_real_fft[: len(joint_pos_real_fft) // 2])

        magnitude_sim_filtered = magnitude_sim[
            joint_pos_sim_fft_freq[: len(joint_pos_sim_fft) // 2] < freq_max
        ]
        magnitude_real_filtered = magnitude_real[
            joint_pos_real_fft_freq[: len(joint_pos_real_fft) // 2] < freq_max
        ]
        error_fft = np.sqrt(
            np.mean((magnitude_real_filtered - magnitude_sim_filtered) ** 2)
        )

        logger.info(
            f"{jnt_name} root mean squared error: {error}, fft error: {error_fft}."
            f"final error: {error + error_fft * 0.01 =:.3f} "
        )
        return error + error_fft * 0.01

    return objective



# per episode contains:
# 1~2 sysID jnts, 1 kp value for all relative motors.
def _optimize_for_one_jnt_with_multiple_episodes(*,
                                                 robot: Robot,
                                                 sim_name: str,
                                                 jnt_name: str,
                                                 ep_list: List[_SysIDEpisodeData],

                                                 # obs_list: List[npt.NDArray[np.float32]],
                                                 # jnt_pos_list: List[float],
                                                 # action_list:
                                                 # motor_act: List[npt.NDArray[np.float32]],
                                                 # motor_kp: Mapping[str, float],

                                                 n_iters: int = 1000,
                                                 early_stop_rounds: int = 200,
                                                 freq_max: float = 10,
                                                 sampler_name: str = "CMA",
                                                 # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
                                                 damping_range: Tuple[float, float, float] = (0.0, 0.5, 1e-3),
                                                 armature_range: Tuple[float, float, float] = (0.0, 0.01, 1e-4),
                                                 frictionloss_range: Tuple[float, float, float] = (0.0, 1.0, 1e-3),
                                                 q_dot_tau_max_range: Tuple[float, float, float] = (0.0, 5.0, 1e-2),
                                                 q_dot_max_range: Tuple[float, float, float] = (5.0, 10.0, 1e-1),
                                                 ) -> Tuple[Dict[str, float], float]:
    """Optimize the parameters of a robot joint using simulation and Optuna.

    This function performs parameter optimization for a specified joint of a robot using a simulation environment. It utilizes Optuna for hyperparameter tuning to minimize the error between simulated and observed joint positions.

    Args:
        robot (Robot): The robot object containing joint and motor information.
        sim_name (str): The name of the simulation environment, currently supports "mujoco".
        jnt_name (str): The name of the joint to optimize.
        ep_list:
        # obs_list (List[npt.NDArray[np.float32]]): List of observed joint positions.
        # action_list (List[npt.NDArray[np.float32]]): List of actions applied to the joint.
        # kp (List[float]): List of proportional gains for the motor.
        n_iters (int, optional): Number of optimization iterations. Defaults to 1000.
        early_stop_rounds (int, optional): Number of rounds for early stopping. Defaults to 200.
        freq_max (float, optional): Maximum frequency for filtering in Fourier Transform. Defaults to 10.
        sampler_name (str, optional): Name of the Optuna sampler to use. Defaults to "CMA".
        damping_range (Tuple[float, float, float], optional): Range for damping parameter. Defaults to (0.0, 0.5, 1e-3).
        armature_range (Tuple[float, float, float], optional): Range for armature parameter. Defaults to (0.0, 0.01, 1e-4).
        frictionloss_range (Tuple[float, float, float], optional): Range for friction loss parameter. Defaults to (0.0, 1.0, 1e-3).
        q_dot_tau_max_range (Tuple[float, float, float], optional): Range for q_dot_tau_max parameter. Defaults to (0.0, 5.0, 1e-2).
        q_dot_max_range (Tuple[float, float, float], optional): Range for q_dot_max parameter. Defaults to (5.0, 10.0, 1e-1).

    Returns:
        Tuple[Dict[str, float], float]: The best parameters found and the corresponding error value.

    Raises:
        ValueError: If an invalid simulator or sampler is specified.
    """

    # if sim_name == "mujoco":
    #     sim = MuJoCoSim(robot, fixed_base=True)
    #
    # else:
    #     raise ValueError("Invalid simulator")

    tau_max_range: Tuple[float, float, float] |None = None
    if "sysID".casefold() in robot.name.casefold():
        tau_max_range: Tuple[float, float, float] = (0.0, 2.0, 1e-2)
        if "XC330".casefold() in robot.name.casefold():
            tau_max_range = (0.0, 1.0, 1e-2)
        elif "XM430".casefold() in robot.name.casefold():
            tau_max_range = (0.0, 3.0, 1e-2)
        elif 'SM40BL'.casefold() in robot.name.casefold():
            # TODO: keep same value as add_default_settings() in process_mjcf.py
            tau_max_range = (0.0, 4.0, 1e-2)

    # motor_names = robot.active_joint_to_motor_name[jnt_name]
    # joint_idx = robot.active_joint_name_ordering.index(jnt_name)

    # concatenate all jnt pos in multiple episode of jnt_name.
    # joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])

    sampler: optuna.samplers.BaseSampler | None = None

    # For RandomSampler, MedianPruner is the best.
    # For TPESampler, HyperbandPruner is the best.

    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler()
    elif sampler_name == "CMA":
        sampler = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("Invalid sampler")

    time_str = time.strftime("%Y%m%d_%H%M%S")

    # TODO: check database table exist.
    storage = "postgresql://optuna_user:password@localhost/optuna_db"

    study = optuna.create_study(
        study_name=f"{robot.name}_{jnt_name}_{time_str}",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    # initial_trial = dict(
    #     damping=float(sim.model.joint(jnt_name).damping[0]),
    #     armature=float(sim.model.joint(jnt_name).armature[0]),
    #     frictionloss=float(sim.model.joint(jnt_name).frictionloss[0]),
    # )
    if "sysID" in robot.name:
        # assert isinstance(sim.controller, MotorController)
        motor_name: List[str] = robot.active_joint_to_motor_name[jnt_name]
        # this joint has more than one corresponding motors, e.g. , waist_roll  -> waist_act_1, waist_act_2.
        if not len(motor_name) == 1:
            raise ValueError(f'TBD: we do not support more than one motor name for sysID.'
                             f'corresponding motor name: {motor_name}')

        motor_idx:int = robot.motor_name_ordering.index(motor_name[0])

        # initial_trial.update(
        #     dict(
        #         tau_max= float(sim.controller.tau_max[motor_idx]),
        #         q_dot_tau_max= float(sim.controller.q_dot_tau_max[motor_idx]),
        #         q_dot_max=float(sim.controller.q_dot_max[motor_idx]),
        #     )
        # )

    # logger.info(f'study enqueue : {initial_trial=:}')
    # study.enqueue_trial(initial_trial)

    objective = _build_objective(
        robot=robot,
        sim=None,  # sim,
        jnt_name=jnt_name,
        ep_list=ep_list,
        freq_max=freq_max,
        damping_range=damping_range,
        armature_range=armature_range,
        frictionloss_range=frictionloss_range,
        q_dot_tau_max_range=q_dot_tau_max_range,
        q_dot_max_range=q_dot_max_range,
        tau_max_range=tau_max_range
    )

    early_stop_cb = _build_early_stop(early_stop_rounds)

    study.optimize(
        objective,
        n_trials=n_iters,  # after optimize, study.trials will include `enqueued` trials.
        # TODO: parrallel possible when using a single Mujoco env?
        n_jobs=12, # 8,  # 1,
        show_progress_bar=True,
        # callbacks=[partial(early_stop_check, early_stopping_rounds=early_stop_rounds)],
        callbacks=[early_stop_cb]
    )

    logger.info(
        f"Opt for sysID joint: {jnt_name} found best parameters: {study.best_params},  "
        f"best value: {study.best_value}"
    )

    sim.close()

    return study.best_params, study.best_value


# def _optimize_all(
#     robot: Robot,
#     sim_name: str,
#     opt_data_gr: Generator[_SysIDJntOptData],
#     # obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
#     # action_dict: Dict[str, List[npt.NDArray[np.float32]]],
#     # kp_dict: Dict[str, List[float]],
#     n_iters: int,
#     early_stop_rounds: int,
#     sysID_jnt_opt_data: Generator[_SysIDJntOptData],
# ):
#     """Optimizes parameters for each joint of a robot using provided observation and action data.
#
#     Args:
#         robot (Robot): The robot instance for which parameters are being optimized.
#         sim_name (str): The name of the simulation.
#         obs_pos_dict (Dict[str, List[npt.NDArray[np.float32]]]): A dictionary mapping joint names to lists of observed positions.
#         action_dict (Dict[str, List[npt.NDArray[np.float32]]]): A dictionary mapping joint names to lists of actions taken.
#         kp_dict (Dict[str, List[float]]): A dictionary mapping joint names to lists of proportional gain values.
#         n_iters (int): The number of iterations for the optimization process.
#         early_stop_rounds (int): The number of rounds for early stopping criteria.
#         sysID_jnt_opt_data: generator to yield sysID data for each sysID jnt in each episode.
#
#     Returns:
#         Tuple[Dict[str, Dict[str, float]], Dict[str, float]]: A tuple containing two dictionaries. The first dictionary maps joint names to their optimized parameters, and the second dictionary maps joint names to their optimized values.
#     """
#
#     # return sysID_file_path
#     # optimize_args: List[
#     #     Tuple[
#     #         Robot,
#     #         str,
#     #         str,
#     #         List[npt.NDArray[np.float32]],
#     #         List[npt.NDArray[np.float32]],
#     #         List[float],
#     #         int,
#     #         int,
#     #     ]
#     # ] = [
#     #     (
#     #         robot,
#     #         sim_name,
#     #         joint_name,
#     #         obs_pos_dict[joint_name],   list of obs pos in multiple episodes of joint_name.
#     #         action_dict[joint_name],    list of action in multiple episodes of joint_name.
#     #         kp_dict[joint_name],        list of kp in multiple episodes of joint_name.
#     #         n_iters,
#     #         early_stop_rounds,
#     #     )
#     #     for joint_name in obs_pos_dict
#     # ]
#
#     # # Create a pool of processes
#     # with Pool(processes=len(obs_pos_dict)) as pool:
#     #     results = pool.starmap(optimize_parameters, optimize_args)
#
#     # # Process results
#     # for joint_name, result in zip(obs_pos_dict.keys(), results):
#     #     opt_params, opt_values = result
#     #     if len(opt_params) > 0:
#     #         opt_params_dict[joint_name] = opt_params
#     #         opt_values_dict[joint_name] = opt_values
#
#     opt_params_dict: Dict[str, Dict[str, float]] = {}
#     opt_values_dict: Dict[str, float] = {}
#     # for args in optimize_args:
#     for _each_jnt_ in sysID_jnt_opt_data:
#         # opt_params, opt_values = _optimize_parameters(*args)
#         # opt_params_dict[args[2]] = opt_params
#         # opt_values_dict[args[2]] = opt_values
#         opt_params, opt_values = _optimize_for_one_jnt_in_one_episode(*args)
#         opt_params_dict[jnt name] = opt_params
#         opt_values_dict[jnt name] = opt_values
#
#     return opt_params_dict, opt_values_dict

def _evaluate(
    robot: Robot,
    sim_name: str,
    sysID_jnt_ep_dict: Dict[str, List[_SysIDEpisodeData]],

    # obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
    # action_dict: Dict[str, List[npt.NDArray[np.float32]]],
    # kp_dict: Dict[str, List[float]],

    opt_params_dict: Dict[str, Dict[str, float]],
    opt_values_dict: Dict[str, float],
    exp_folder: Path,
):
    """Evaluates the performance of a robot simulation by comparing simulated and real joint positions, and logs the results.

    Args:
        robot (Robot): The robot object containing joint and motor configurations.
        sim_name (str): The name of the simulator to use, e.g., "mujoco".
        obs_pos_dict (Dict[str, List[npt.NDArray[np.float32]]]): Dictionary mapping joint names to lists of observed position arrays.
        action_dict (Dict[str, List[npt.NDArray[np.float32]]]): Dictionary mapping joint names to lists of action arrays.
        kp_dict (Dict[str, List[float]]): Dictionary mapping joint names to lists of proportional gain values.
        opt_params_dict (Dict[str, Dict[str, float]]): Dictionary of optimized parameters for each joint.
        opt_values_dict (Dict[str, float]): Dictionary of optimized values for each joint.
        exp_folder (Path): Path to the folder where experiment results will be saved.

    Raises:
        ValueError: If an invalid simulator name is provided.
    """

    opt_params_file = exp_folder /"opt_params.json"
    opt_values_file = exp_folder /"opt_values.json"

    with open(opt_params_file, "wt") as f:
        json.dump(opt_params_dict, f, indent=4)

    with open(opt_values_file, "wt") as f:
        json.dump(opt_values_dict, f, indent=4)

    # dyn_config_file = Path("toddlerbot") / "descriptions"/ robot.name / "config_dynamics.json"
    dyn_config_file = Path("toddlerbot") / "descriptions"/ robot.name / "sysID_dynamics.json"
    if dyn_config_file.exists():
        with open(dyn_config_file, "rt") as _f:
            dyn_config = json.load(_f)

        for _jnt_name in opt_params_dict:
            for param_name in opt_params_dict[_jnt_name]:
                dyn_config[_jnt_name][param_name] = opt_params_dict[_jnt_name][param_name ]
    else:
        dyn_config = opt_params_dict

    with open(dyn_config_file, "wt") as _f:
        _f.writable('//NOTE: this file is auto-generated by sysID_opt.py, do not change it.')
        json.dump(dyn_config, _f, indent=4)

    time_seq_ref_dict: Dict[str, List[float]] = {}
    time_seq_sim_dict: Dict[str, List[float]] = {}
    time_seq_real_dict: Dict[str, List[float]] = {}
    jnt_pos_sim_dict: Dict[str, List[float]] = {}
    jnt_pos_real_dict: Dict[str, List[float]] = {}
    action_sim_dict: Dict[str, List[float]] = {}
    action_real_dict: Dict[str, List[float]] = {}

    # for joint_name in obs_pos_dict:
    for _jnt_name, _ep_list in sysID_jnt_ep_dict.items():
        # obs_list = obs_pos_dict[joint_name]
        # action_list = action_dict[joint_name]
        # kp_list = kp_dict[joint_name]

        jnt_ordering_idx = robot.active_joint_name_ordering.index(_jnt_name)
        jnt_pos_real_arr: npt.NDArray[np.float32] = np.concatenate([_ep.sysID_jnt_pos for _ep in _ep_list])

        # motor_names = robot.active_joint_to_motor_name[joint_name]
        # joint_idx = robot.active_joint_name_ordering.index(joint_name)
        # joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])

        if sim_name == "mujoco":
            sim = MuJoCoSim(robot, fixed_base=True)
        else:
            raise ValueError("Invalid simulator")

        joint_dyn = {
            _jnt_name: {
                "damping": opt_params_dict[_jnt_name]["damping"],
                "armature": opt_params_dict[_jnt_name]["armature"],
                "frictionloss": opt_params_dict[_jnt_name]["frictionloss"],
            }
        }
        sim.set_joint_dynamics(joint_dyn)

        # TODO: for SysID, wo only allow one motor now.
        motor_name: List[str] = robot.active_joint_to_motor_name[_jnt_name]
        # this joint has more than one corresponding motors, e.g. , waist_roll  -> waist_act_1, waist_act_2.
        if not len(motor_name) == 1:
            raise ValueError(f'TBD: we do not support more than one motor name for sysID.'
                             f'corresponding motor name: {motor_name}')

        if "sysID" in robot.name:
            # TODO: for SysID, wo only allow one motor now.
            assert len(motor_name) == 1
            sim.set_motor_dynamics(
                { motor_name[0]: dict(
                    tau_max= opt_params_dict[_jnt_name]["tau_max"],
                    q_dot_tau_max= opt_params_dict[_jnt_name]["q_dot_tau_max"],
                    q_dot_max= opt_params_dict[_jnt_name]["q_dot_max"] )
                }
            )

        # jnt_pos_sim_list: List[npt.NDArray[np.float32]] = []

        # for action, kp in zip(action_list, kp_list):
        #     sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))
        #     for a in action:
        #         obs = sim.get_observation(1)
        #         sim.set_motor_target(a)
        #         sim.step()
        #
        #         assert obs.joint_pos is not None
        #         joint_pos_sim_list.append(obs.joint_pos[joint_idx])
        #
        # joint_pos_sim = np.array(joint_pos_sim_list)

        # jnt_pos_sim_list: List[float] = []
        # each `action` is a sequence belongs to one episode with a kp value.
        # for action, kp in zip(action_list, kp):
        # for _ep in _ep_list:
        #     # TODO: why not adjust joint pos in Mujoco at beginning of each episode ?
        #     # sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))
        #     sim.set_motor_kps(_ep.motor_kp)
        #     logger.info(f'set jnt: {_jnt_name} kp: {_ep.motor_kp}')
        #
        #     # for a in action:
        #     for _act in _ep.motor_act:
        #         obs = sim.get_observation(1)
        #         sim.set_motor_target(_act)
        #         sim.step()
        #
        #         assert obs.joint_pos is not None
        #         jnt_pos_sim_list.append(obs.joint_pos[jnt_ordering_idx])
        #
        # jnt_pos_sim_arr = np.asarray(jnt_pos_sim_list, dtype=np.float32)
        # logger.info(f'jnt: {_jnt_name} joint_pos_sim_arr shape:{jnt_pos_sim_arr.shape}')

        jnt_pos_sim_arr: npt.NDArray[np.float32] = _sim_run_sysID_episode_helper(ep_list=_ep_list,
                                                                                 jnt_name=_jnt_name,
                                                                                 jnt_ordering_idx=jnt_ordering_idx,
                                                                                 sim=sim,
                                                                                 )
        assert jnt_pos_real_arr.shape == jnt_pos_sim_arr.shape

        error = np.sqrt(np.mean( (jnt_pos_real_arr - jnt_pos_sim_arr) ** 2) )

        logger.info(
            f"{_jnt_name} root mean squared error: {error}"
        )

        time_seq_ref_dict[_jnt_name] = (
            # np.arange(sum([len(action) for action in action_list]))
            ( np.arange(sum( len(_ep.sysID_jnt_pos) for _ep in _ep_list ))
            * (sim.n_frames * sim.dt) ).tolist()
        )

        time_seq_sim_dict[_jnt_name] = time_seq_ref_dict[_jnt_name]
        time_seq_real_dict[_jnt_name] = time_seq_ref_dict[_jnt_name]

        jnt_pos_sim_dict[_jnt_name] = jnt_pos_sim_arr.tolist()
        jnt_pos_real_dict[_jnt_name] = jnt_pos_real_arr.tolist()


        # motor_name : List[str] = robot.active_joint_to_motor_name[_jnt_name]
        # # this joint has more than one corresponding motors, e.g. , waist_roll  -> waist_act_1, waist_act_2.
        # if not len(motor_name) == 1:
        #     raise ValueError(f'TBD: we do not support more than one motor name for plot joint track.'
        #                      f'corresponding motor name: {motor_name}')

        motor_ordering_idx = robot.motor_name_ordering.index(motor_name[0])
        motor_act_real_list:List[float] = []
        for _ep in _ep_list:
            motor_act_real_list.extend( _a[motor_ordering_idx] for _a in _ep.motor_act )

        # motor_act_real_arr = np.concatenate(
            # [action[:, jnt_ordering_idx] for action in action_list]
        # ).tolist()

        # action_sim_dict[_jnt_name] = action_all
        # action_real_dict[_jnt_name] = action_all

        action_sim_dict[_jnt_name] = motor_act_real_list
        action_real_dict[_jnt_name] = motor_act_real_list

        sim.close()

    plot_save_folder:str = (exp_folder / 'plot').resolve().__str__()

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        jnt_pos_sim_dict,
        jnt_pos_real_dict,
        robot.joint_cfg_limits,
        save_path=plot_save_folder,
        file_name="sim2real_joint_pos",
        line_suffix=["_sim", "_real"],
    )
    plot_joint_tracking_frequency(
        time_seq_sim_dict,
        time_seq_real_dict,
        jnt_pos_sim_dict,
        jnt_pos_real_dict,
        save_path=plot_save_folder,
        file_name="sim2real_joint_freq",
        line_suffix=["_sim", "_real"],
    )
    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_ref_dict,
        jnt_pos_sim_dict,
        action_sim_dict,
        robot.joint_cfg_limits,
        save_path=plot_save_folder,
        file_name="sim_tracking",
    )
    plot_joint_tracking_frequency(
        time_seq_sim_dict,
        time_seq_ref_dict,
        jnt_pos_sim_dict,
        action_sim_dict,
        save_path=plot_save_folder,
        file_name="sim_tracking_freq",
    )
    plot_joint_tracking(
        time_seq_real_dict,
        time_seq_ref_dict,
        jnt_pos_real_dict,
        action_real_dict,
        robot.joint_cfg_limits,
        save_path=plot_save_folder,
        file_name="real_tracking",
    )
    plot_joint_tracking_frequency(
        time_seq_real_dict,
        time_seq_ref_dict,
        jnt_pos_real_dict,
        action_real_dict,
        save_path=plot_save_folder,
        file_name="real_tracking_freq",
    )


def _main(args: argparse.Namespace):
    """Executes the SysID optimization process for a specified robot and simulator.

    This function parses command-line arguments to configure the optimization process,
    validates the experiment folder path, and initializes the robot and experiment settings.
    It then loads datasets, optimizes hyperparameters, and evaluates the optimized parameters
    in the simulation.

    Raises:
        ValueError: If the specified experiment folder path does not exist.
    """

    # data_path = Path("results") / f'{args.robot}_{args.policy}_real_world_{args.time_str} '
    # data_folder:Path = Path( RUN_POLICY_LOG_FOLDER_FMT.format(robot_name=args.robot,
    #                                                   policy_name=args.policy,
    #                                                   env_name='real_world',
    #                                                   cur_time=args.time_str))
    data_folder: Path = Path(args.data_folder)

    step_record_file: Path = data_folder / RUN_STEP_RECORD_PICKLE_FILE
    kp_file: Path = data_folder/ RUN_EPISODE_MOTOR_KP_PICKLE_FILE #  .format(policy_name = args.policy)

    if not step_record_file.exists():
        raise ValueError(f"Invalid step record file path: {step_record_file.resolve()}")
    if not kp_file.exists():
        raise ValueError(f"Invalid kp file path: {kp_file.resolve()}")

    robot = Robot(args.robot)

    exp_name = f'{robot.name}_sysID_{args.sim}_optim'
    time_str = time.strftime('%Y%m%d_%H%M%S')
    exp_folder = Path(f'run_sysID/{exp_name}_{time_str}')

    exp_folder.mkdir(parents=True, exist_ok=True)

    with open( exp_folder/'opt_config.json', 'wt') as _f:
        json.dump(vars(args), _f, indent=4)

    # obs_pos_dict, action_dict, kp_dict = _load_dataset(robot, step_record_file, kp_file)
    # sysID_jnt_opt_data :Generator[_SysIDJntOptData] = _load_dataset(robot=robot, step_record_file=step_record_file, kp_file=kp_file)

    # one sysID jnt: multiple episodes.
    sysID_jnt_ep_dict: Dict[str, List[_SysIDEpisodeData]] = defaultdict(list)

    # TODO: copy _ep_data ?
    for _ep_data in _load_dataset(robot=robot, step_record_file=step_record_file, kp_file=kp_file):
        sysID_jnt_ep_dict[_ep_data.sysID_jnt_name].append(_ep_data)

    logger.info(f'load all episode data from pickle file:')
    for _k,_v in sysID_jnt_ep_dict.items():
        logger.info(f'jnt name:{_k}  number of episodes: {len(_v)}')

    ###### Optimize the hyperparameters ######
    # optimize_parameters(
    #     robot,
    #     args.sim,
    #     "waist_yaw",
    #     obs_pos_dict["waist_yaw"],
    #     action_dict["waist_yaw"],
    #     args.n_iters,
    # )

    opt_params_dict: Dict[str, Dict[str, float]] = {}

    opt_values_dict: Dict[str, float] = {}

    # _jnt_name : single sysID_jnt,  _ep_list:  all the episodes belong to the single sysID_jnt.
    for _jnt_name, _ep_list in sysID_jnt_ep_dict.items():
        # opt_params, opt_values = _optimize_parameters(*args)
        # opt_params_dict[args[2]] = opt_params
        # opt_values_dict[args[2]] = opt_values

        # optimize using all episodes of a singel sysID_jnt:
        opt_params, opt_values = _optimize_for_one_jnt_with_multiple_episodes(
            robot=robot,
            sim_name=args.sim,
            jnt_name=_jnt_name,
            ep_list=_ep_list,
            n_iters=args.n_iters,
            early_stop_rounds=args.early_stop,
        )
        assert _jnt_name not in opt_params_dict
        assert _jnt_name not in opt_values_dict

        opt_params_dict[_jnt_name] = opt_params
        opt_values_dict[_jnt_name] = opt_values

    # opt_params_dict, opt_values_dict = _optimize_all(
    #     robot,
    #     args.sim,
    #     # obs_pos_dict,
    #     # action_dict,
    #     # kp_dict,
    #     args.n_iters,
    #     args.early_stop,
    #     sysID_jnt_opt_data
    # )

    ##### Evaluate the optimized parameters in the simulation ######
    _evaluate(
        robot,
        args.sim,
        sysID_jnt_ep_dict,
        # obs_pos_dict,
        # action_dict,
        # kp_dict,
        opt_params_dict,
        opt_values_dict,
        exp_folder,
    )

def _args_parsing() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run the SysID optimization.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="sysID_fixed",
        help="The name of the task.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=500,
        help="The number of iterations to optimize the parameters.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=200,
        help="The number of iterations to early stop the optimization.",
    )
    # parser.add_argument(
    #     "--time-str",
    #     type=str,
    #     default="",
    #     required=True,
    #     dest='time_str',
    #     help="The name of the run.",
    # )

    parser.add_argument(
        "--data-folder",
        type=str,
        default="",
        required=True,
        help="The name of the data folder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    _parsed_args = _args_parsing()
    # TODO: move into yaml config.
    config_logging(root_logger_level=logging.INFO, root_handler_level=logging.NOTSET,
                   root_fmt='--- {levelname} - module:{module} - func:{funcName} ---> \n{message}',
                   root_date_fmt='%Y-%m-%d %H:%M:%S',
                   # log_file='/tmp/toddler/imitate_episode.log',
                   log_file=None,
                   module_logger_config={'tools': logging.INFO,
                                         'main': logging.INFO})
    # use root logger for __main__.
    logger = logging.getLogger('root')
    logger.info('parsed args --->\n{}'.format('\n'.join(
        f'{arg_name}={arg_value}' for arg_name, arg_value in
        sorted(_parsed_args.__dict__.items(), key=lambda k_v_pair: k_v_pair[0]))))

    _main(_parsed_args)

