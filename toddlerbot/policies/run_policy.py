import argparse
# import bisect
from importlib import import_module
import json
from pathlib import Path
import pickle
# import pkgutil
import time
import time as timelib
from typing import (Dict, List, Optional, Generator,
                    DefaultDict, Mapping, Type,
                    NamedTuple, Set)
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from moviepy import ImageSequenceClip
from tqdm import tqdm
import logging
from copy import deepcopy

from toddlerbot.sim import (Robot, BaseEnv, MuJoCoSim, RealWorld)
from toddlerbot.utils import ( sync_time, dump_profiling_data,
                               # snake2camel,
                               config_logging,
                               # profile
                               )
from toddlerbot.visualization import *
from toddlerbot.policies.base_policy import BasePolicy
from toddlerbot.policies._module_logger import logger

# import from ./__init__.py
from toddlerbot.policies import (RUN_EPISODE_MOTOR_KP_PICKLE_FILE,RUN_STEP_RECORD_PICKLE_FILE,
               RUN_POLICY_LOG_FOLDER_FMT,StepRecord)


class _ModuleAndClsName(NamedTuple):
    module_name : str   # python file name.
    cls_name: str

# TODO: complete this.
_policy_module_and_cls_dict :Dict[str, _ModuleAndClsName ]={
    'balance_pd': _ModuleAndClsName('balance_pd', 'BalancePDPolicy'),
    'sysID': _ModuleAndClsName('sysID', 'SysIDPolicy'),
    'calibrate': _ModuleAndClsName ('calibrate', 'CalibratePolicy', ),
}


def _get_policy_class_v2(policy_name: str) -> Type["BasePolicy"]:
    """Retrieves the policy class associated with the given policy name.

    Args:
        policy_name (str): The name of the policy to retrieve.

    Returns:
        Type[BasePolicy]: The class of the policy corresponding to the given name.

    Raises:
        ValueError: If the policy name is not found in the policy registry.
    """
    if policy_name not in _policy_module_and_cls_dict:
        raise ValueError(f"Unknown policy: {policy_name}")

    mod_name: str = _policy_module_and_cls_dict[policy_name].module_name
    cls_name: str = _policy_module_and_cls_dict[policy_name].cls_name

    logger.info(f'dynamically import module:{mod_name}, class name: {cls_name}')
    module = import_module('.' + mod_name,
                           'toddlerbot.policies.implementations' )

    if hasattr(module, cls_name):
        return getattr(module, cls_name)
    else:
        raise ValueError(f'imported module: {mod_name} has no attr with name: {cls_name}.'
                         f' module dict: {module.__dict__} ' )


def _get_policy_names_v2() -> Set[str]:
    """Retrieves a list of policy names from the policy registry.

    This function iterates over the keys in the policy registry and generates a list
    of policy names. For each key, it adds the key itself and a modified version of
    the key with the suffix '_fixed' to the list.

    Returns:
        List[str]: A list containing the original and modified policy names.
    """
    return  (  {_k for _k in _policy_module_and_cls_dict } |
               {_k + "_fixed" for _k in _policy_module_and_cls_dict }
               )

# def dynamic_import_policies(policy_package: str):
#     """Dynamically imports all modules within a specified package.
#
#     This function attempts to import each module found in the given package directory. If a module cannot be imported, a log message is generated.
#
#     Args:
#         policy_package (str): The name of the package containing the modules to be imported.
#     """
#     package = importlib.import_module(policy_package)
#     package_path = package.__path__
#
#     # Iterate over all modules in the given package directory
#     for _, module_name, _ in pkgutil.iter_modules(package_path):
#         full_module_name = f"{policy_package}.{module_name}"
#         try:
#             importlib.import_module(full_module_name)
#         except Exception as err:
#             logger.error(f"Could not import {full_module_name}, err:{err}")


# Call this to import all policies dynamically
# dynamic_import_policies("toddlerbot.policies")


def _plot_loop_time_helper(step_record_list: List[StepRecord], plot_dir: Path):
    # loop_time_dict: Dict[str, List[float]] = {
    #     "obs_time": [],
    #     "inference": [],
    #     "set_action": [],
    #     "sim_step": [],
    #     "log_time": [],
    #     # "total_time": [],
    # }
    loop_time_dict: DefaultDict[str, List[float]] = defaultdict(list)
    # for i, loop_time in enumerate(loop_time_list):
    for _r in step_record_list:
        # (
        #     step_start,
        #     obs_time,
        #     inference_time,
        #     set_action_time,
        #     sim_step_time,
        #     step_end,
        # ) = _r.time_pnt
        t = _r.time_pnt
        loop_time_dict["obs_time"].append((t.recv_obs - t.step_start) * 1000)
        loop_time_dict["inference"].append((t.inference - t.recv_obs) * 1000)
        loop_time_dict["set_action"].append((t.set_action - t.inference) * 1000)
        loop_time_dict["sim_step"].append((t.sim_step - t.set_action) * 1000)
        loop_time_dict["log_time"].append((t.step_end - t.sim_step) * 1000)
        # loop_time_dict["total_time"].append((step_end - step_start) * 1000)

    plot_loop_time(loop_time_dict=loop_time_dict, save_path=plot_dir.resolve().__str__())
    del loop_time_dict


def _plot_obs_helper(*, robot: Robot,
                     time_obs_list: List[float],
                     ang_vel_obs_list: List[npt.NDArray[np.float32]],
                     euler_obs_list: List[npt.NDArray[np.float32]],
                     tor_obs_total_list: List[float],
                     motor_vel_dict: Mapping[str, List[float]],
                     motor_tor_dict: Mapping[str, List[float]],
                     plot_dir:Path
                     ):
    if "sysID" in robot.name:
        plot_motor_vel_tor_mapping(
            motor_vel_dict["joint_0"],
            motor_tor_dict["joint_0"],
            save_path=plot_dir.resolve().__str__(),
        )

    plot_line_graph(
        tor_obs_total_list,
        time_obs_list,
        legend_labels=["Torque (Nm) or Current (mA)"],
        title="Total Torque or Current  Over Time",
        x_label="Time (s)",
        y_label="Torque (Nm) or Current (mA)",
        save_config=True,
        save_path=plot_dir.resolve().__str__(),
        file_name="total_tor_tracking",
    )()
    plot_line_graph(
        np.array(ang_vel_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Angular Velocities Over Time",
        x_label="Time (s)",
        y_label="Angular Velocity (rad/s)",
        save_config=True,
        save_path=plot_dir.resolve().__str__(),
        file_name="ang_vel_tracking",
    )()
    plot_line_graph(
        np.array(euler_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Euler Angles Over Time",
        x_label="Time (s)",
        y_label="Euler Angles (rad)",
        save_config=True,
        save_path=plot_dir.resolve().__str__(),
        file_name="euler_tracking",
    )()

    # if len(control_inputs_dict) > 0:
    #     plot_path_tracking(
    #         time_obs_list,
    #         pos_obs_list,
    #         euler_obs_list,
    #         control_inputs_dict,
    #         save_path=exp_folder.resolve().__str__(),
    #     )


def _plot_action_helper(*, robot:Robot,
                        time_seq_dict: Dict[str, List[float]],
                        time_seq_ref_dict: Dict[str, List[float]],
                        motor_pos_dict: Dict[str, List[float]],
                        motor_vel_dict: Dict[str, List[float]],
                        motor_tor_dict: Dict[str, List[float]],
                        action_dict: Dict[str, List[float]],
                        plot_dir: Path
                        ):
    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        robot.joint_cfg_limits,
        save_path=plot_dir.resolve().__str__(),
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_tor_dict,
        save_path=plot_dir.resolve().__str__(),
        y_label="Torque (Nm) or Current (mA)",
        file_name="motor_tor_tracking",
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_vel_dict,
        save_path=plot_dir.resolve().__str__(),
    )
    plot_joint_tracking_frequency(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        save_path=plot_dir.resolve().__str__(),
    )


def _plot_run_log(
    robot: Robot,
    step_record_list: List[StepRecord],
    # loop_time_list: List[List[float]],
    # obs_list: List[Obs],
    # control_inputs_list: List[Mapping[str, float]],
    # motor_angles_list: List[Mapping[str, float]],
    plot_dir: Path
):
    """Generates and saves various plots to visualize the performance and behavior of a robot during an experiment.

    Args:
        robot (Robot): The robot object containing information about the robot's configuration and state.
        step_record_list:
        # loop_time_list (List[List[float]]): A list of lists containing timing information for each loop iteration.
        # obs_list (List[Obs]): A list of observations recorded during the experiment.
        # control_inputs_list (List[Mapping[str, float]]): A list of dictionaries containing control inputs applied to the robot.
        # motor_angles_list (List[Mapping[str, float]]): A list of dictionaries containing motor angles recorded during the experiment.
        plot_dir (Path): The path to the folder where the plots will be saved.
    """
    plt.switch_backend("Agg")

    if not plot_dir.exists():
        plot_dir.mkdir()

    _plot_loop_time_helper(step_record_list, plot_dir)

    time_obs_list: List[float] = []
    # lin_vel_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    pos_obs_list: List[npt.NDArray[np.float32]] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    tor_obs_total_list: List[float] = []
    # time_seq_dict: Mapping[str, List[float]] = {}
    # time_seq_ref_dict: Mapping[str, List[float]] = {}
    # motor_pos_dict: Mapping[str, List[float]] = {}
    # motor_vel_dict: Mapping[str, List[float]] = {}
    # motor_tor_dict: Mapping[str, List[float]] = {}
    time_seq_dict: DefaultDict[str, List[float]] = defaultdict(list)
    time_seq_ref_dict: DefaultDict[str, List[float]] = defaultdict(list)
    motor_pos_dict: DefaultDict[str, List[float]] = defaultdict(list)
    motor_vel_dict: DefaultDict[str, List[float]] = defaultdict(list)
    motor_tor_dict: DefaultDict[str, List[float]] = defaultdict(list)

    # for i, obs in enumerate(obs_list):
    for _r in step_record_list:
        obs = _r.obs
        time_obs_list.append(obs.time)
        # lin_vel_obs_list.append(obs.lin_vel)
        # IMU state.
        ang_vel_obs_list.append(obs.ang_vel)
        pos_obs_list.append(obs.pos)
        euler_obs_list.append(obs.euler)
        tor_obs_total_list.append(sum(obs.motor_tor))

        # motor state.
        for _x, _n in enumerate(robot.motor_name_ordering):
            # if _n not in time_seq_dict:
            #     time_seq_ref_dict[_n] = []
            #     time_seq_dict[_n] = []
            #     motor_pos_dict[_n] = []
            #     motor_vel_dict[_n] = []
            #     motor_tor_dict[_n] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[_n].append(obs.time)
            time_seq_ref_dict[_n].append(obs.time)
            # time_seq_ref_dict[_n].append(i * policy.control_dt)
            motor_pos_dict[_n].append(obs.motor_pos[_x])
            motor_vel_dict[_n].append(obs.motor_vel[_x])
            motor_tor_dict[_n].append(obs.motor_tor[_x])

    # action_dict: Dict[str, List[float]] = {}
    action_dict: DefaultDict[str, List[float]] = defaultdict(list)
    # joint_pos_ref_dict: Dict[str, List[float]] = {}
    active_joint_pos_ref_dict: DefaultDict[str, List[float]] = defaultdict(list)
    # for motor_angles in motor_angles_list:
    for _r in step_record_list:
        motor_act_arr: npt.NDArray[np.float32] = _r.motor_act
        named_motor_act: OrderedDict[str, float] = OrderedDict()
        # for motor_name, motor_angle in motor_angles.items():
        #     if motor_name not in action_dict:
        #         action_dict[motor_name] = []
        #     action_dict[motor_name].append(motor_angle)

        for _x, _n in enumerate(robot.motor_name_ordering):
            action_dict[_n].append(motor_act_arr[_x])
            named_motor_act[_n] = motor_act_arr[_x]

        for _n, _a in robot.motor_to_active_joint_angles(named_motor_act).items():
            active_joint_pos_ref_dict[_n].append(_a)

        # joint_angle_ref = robot.motor_to_active_joint_angles(motor_angles)
        # for joint_name, joint_angle in joint_angle_ref.items():
        #     if joint_name not in joint_pos_ref_dict:
        #         joint_pos_ref_dict[joint_name] = []
        #     joint_pos_ref_dict[joint_name].append(joint_angle)

    # control_inputs_dict: Dict[str, List[float]] = {}
    ctrl_inputs_dict: DefaultDict[str, List[float]] = defaultdict(list)
    # for control_inputs in control_inputs_list:
    for _r in step_record_list:
        # for control_name, control_input in control_inputs.items():
        #     if control_name not in control_inputs_dict:
        #         control_inputs_dict[control_name] = []
        #     control_inputs_dict[control_name].append(control_input)
        for _n, _c in _r.ctrl_input.items():
            ctrl_inputs_dict[_n].append(_c)


    _plot_obs_helper(robot=robot,
                     time_obs_list=time_obs_list,
                     ang_vel_obs_list=ang_vel_obs_list,
                     euler_obs_list=euler_obs_list,
                     tor_obs_total_list=tor_obs_total_list,
                     motor_vel_dict=motor_vel_dict,
                     motor_tor_dict=motor_tor_dict,
                     plot_dir=plot_dir)

    _plot_action_helper(robot=robot,
                        time_seq_dict=time_seq_dict,
                        time_seq_ref_dict=time_seq_ref_dict,
                        motor_pos_dict=motor_pos_dict,
                        motor_vel_dict=motor_vel_dict,
                        motor_tor_dict=motor_tor_dict,
                        action_dict=action_dict,
                        plot_dir=plot_dir)

    # plt.switch_backend("Agg")
    #
    # plot_loop_time(loop_time_dict, exp_folder_path)

    # if "sysID" in robot.name:
    #     plot_motor_vel_tor_mapping(
    #         motor_vel_dict["joint_0"],
    #         motor_tor_dict["joint_0"],
    #         save_path=exp_folder.resolve().__str__(),
    #     )

    # if hasattr(policy, "com_pos_list"):
    #     plot_len = min(len(policy.com_pos_list), len(time_obs_list))
    #     plot_line_graph(
    #         np.array(policy.com_pos_list).T[:2, :plot_len],
    #         time_obs_list[:plot_len],
    #         legend_labels=["COM X", "COM Y"],
    #         title="Center of Mass Over Time",
    #         x_label="Time (s)",
    #         y_label="COM Position (m)",
    #         save_config=True,
    #         save_path=exp_folder,
    #         file_name="com_tracking",
    #     )()

    # plot_line_graph(
    #     tor_obs_total_list,
    #     time_obs_list,
    #     legend_labels=["Torque (Nm) or Current (mA)"],
    #     title="Total Torque or Current  Over Time",
    #     x_label="Time (s)",
    #     y_label="Torque (Nm) or Current (mA)",
    #     save_config=True,
    #     save_path=exp_folder.resolve().__str__(),
    #     file_name="total_tor_tracking",
    # )()
    # plot_line_graph(
    #     np.array(ang_vel_obs_list).T,
    #     time_obs_list,
    #     legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
    #     title="Angular Velocities Over Time",
    #     x_label="Time (s)",
    #     y_label="Angular Velocity (rad/s)",
    #     save_config=True,
    #     save_path=exp_folder.resolve().__str__(),
    #     file_name="ang_vel_tracking",
    # )()
    # plot_line_graph(
    #     np.array(euler_obs_list).T,
    #     time_obs_list,
    #     legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
    #     title="Euler Angles Over Time",
    #     x_label="Time (s)",
    #     y_label="Euler Angles (rad)",
    #     save_config=True,
    #     save_path=exp_folder.resolve().__str__(),
    #     file_name="euler_tracking",
    # )()
    # # if len(control_inputs_dict) > 0:
    # #     plot_path_tracking(
    # #         time_obs_list,
    # #         pos_obs_list,
    # #         euler_obs_list,
    # #         control_inputs_dict,
    # #         save_path=exp_folder.resolve().__str__(),
    # #     )
    # plot_joint_tracking(
    #     time_seq_dict,
    #     time_seq_ref_dict,
    #     motor_pos_dict,
    #     action_dict,
    #     robot.joint_cfg_limits,
    #     save_path=exp_folder.resolve().__str__(),
    # )
    # plot_joint_tracking_single(
    #     time_seq_dict,
    #     motor_tor_dict,
    #     save_path=exp_folder.resolve().__str__(),
    #     y_label="Torque (Nm) or Current (mA)",
    #     file_name="motor_tor_tracking",
    # )
    # plot_joint_tracking_single(
    #     time_seq_dict,
    #     motor_vel_dict,
    #     save_path=exp_folder.resolve().__str__(),
    # )
    # plot_joint_tracking_frequency(
    #     time_seq_dict,
    #     time_seq_ref_dict,
    #     motor_pos_dict,
    #     action_dict,
    #     save_path=exp_folder.resolve().__str__(),
    # )



@dataclass(init=True)
class _MotorKpSetter:
    _cur_ep_idx:int = -1

    def set_kp(self, *,
                     policy:BasePolicy, env:BaseEnv, step_count:int, obs_time: float):
        # assert isinstance(policy, SysIDPolicy)
        assert type(policy).__name__ ==  'SysIDPolicy'
        # key is end_time of each episode.
        # ep_end_time_pnt = tuple(policy.episode_motor_kp)  #.keys())
        # if obs.time > max(ep_end_time_point), bisect_left will return len(ep_end_time_point) as `insertion` position.
        # ep_idx = min(bisect.bisect_left(ep_end_time_pnt, obs.time),
        #              len(ep_end_time_pnt) - 1)

        # if ep_idx != prev_ep_idx:
        # motor_kps = policy.episode_motor_kp[ep_end_time_pnt[ep_idx]]
        #     motor_kps_updated = {}
        #     for _jnt_name in motor_kps:
        #         for motor_name in robot.active_joint_to_motor_name[_jnt_name]:
        #             motor_kps_updated[motor_name] = motor_kps[_jnt_name]
        #
        #     # not update if zero.
        #     if np.any(list(motor_kps_updated.values())):
        #         env.set_motor_kps(motor_kps_updated)
        #         prev_ep_idx = ep_idx

        # policy.episode_motor_kp is ordered list.

        # always set first ep kp.
        if step_count == 0 or self._cur_ep_idx==-1:
            # we do not allow obs skip an episode.
            assert obs_time <= policy.episode_info[0].ep_end_time_pnt
            self._cur_ep_idx = 0

            env.set_motor_kps(policy.episode_info[self._cur_ep_idx].motor_kp)

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

            # TODO: if all kp are zero, not set ,keep kp value set previously? but episode_motor_kp
            # will be recorded into log_data_dict....
            # not update if all zero. but motor is already set previous kp value, no change?
            # if np.any(tuple(motor.values())):
            #     env.set_motor_kps(motor_kps_updated)
            #     prev_ep_idx = ep_idx

            env.set_motor_kps(policy.episode_info[self._cur_ep_idx].motor_kp)

            logger.info(f'update cur episode idx to {self._cur_ep_idx}, '
                        f'and set motor kp: {policy.episode_info[self._cur_ep_idx].motor_kp}')

        else:
            # kp no change, not set.
            pass



def _toggle_motor(policy:BasePolicy, env:BaseEnv):
    # need to enable and disable motors according to logging state
    # if isinstance(policy, TeleopLeaderPolicy) and policy.toggle_motor:
    if type(policy).__name__ =='TeleopLeaderPolicy' and policy.toggle_motor:
        assert isinstance(env, RealWorld)
        if policy.is_running:
            # disable all motors when logging
            env.actuator_controller.disable_motors()
        else:
            # enable all motors when not logging
            env.actuator_controller.enable_motors()

        policy.toggle_motor = False

    # elif isinstance(policy, RecordPolicy) and policy.toggle_motor:
    elif type(policy).__name__ =='RecordPolicy' and policy.toggle_motor:
        assert isinstance(env, RealWorld)
        env.actuator_controller.disable_motors(policy.disable_motor_indices)
        policy.toggle_motor = False


def _save_run_log(step_record_list: List[StepRecord], pickle_file: Path):
    # log_dir = exp_folder / 'step_record'
    if not pickle_file.parent.exists():
        pickle_file.parent.mkdir(parents=True)

    # with open(log_dir / 'step_record_list.pkl', 'wb') as _f:
    with open(pickle_file, 'wb') as _f:
        pickle.dump(step_record_list, _f)


def _save_policy_log(*, policy:BasePolicy, robot:Robot,
                     log_dir: Path,
                     step_record_list: List[StepRecord]):
    # log_dir = exp_folder / policy.name
    if not log_dir.exists():
        log_dir.mkdir()

    # if isinstance(policy, SysIDPolicy):
    if type(policy).__name__ == 'SysIDPolicy':
        # with open(log_dir/'episode_motor_kp.pkl', "wb") as _f:
        with open(log_dir / RUN_EPISODE_MOTOR_KP_PICKLE_FILE,  # .format(policy_name=policy.name),
                  "wb") as _f:
            pickle.dump(policy.episode_info, _f)

    # if isinstance(policy, TeleopFollowerPDPolicy):
    if type(policy).__name__ == 'TeleopFollowerPDPolicy':
        policy.dataset_logger.move_files_to_folder(log_dir)

    # if isinstance(policy, DPPolicy) and len(policy.camera_frame_list) > 0:
    if type(policy).__name__ == 'DPPolicy' and len(policy.camera_frame_list) > 0:
        fps = int(1 / np.diff(policy.camera_time_list).mean())
        logger.info(f"visual_obs fps: {fps}")
        video_path = log_dir / "visual_obs.mp4"
        video_clip = ImageSequenceClip(policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    # if isinstance(policy, ReplayPolicy):
    if type(policy).__name__ == 'ReplayPolicy':
        with open(log_dir/ 'keyframes.pkl', 'wb') as _f:
            pickle.dump(policy.keyframes, _f)

    # if isinstance(policy, CalibratePolicy):
    if type(policy).__name__ == 'CalibratePolicy':
        # TODO: after run_policy, call add_configs.py to update config_motors.json contents into config.json.
        config_file: Path = robot.root_path / 'joint_motor_mapping.json'
        if config_file.exists():
            # motor_names = robot.get_joint_config_attrs("is_passive", False)
            # motor_pos_init = np.array(robot.get_joint_config_attrs("is_passive", False, "init_pos"),
            #                           dtype=np.float32)

            assert len(step_record_list[-1].motor_act ) == len(policy.default_motor_pos)

            # last action of CalibratePolicy is the action to make robot `Stand` stable and nice, and
            # has slight bias against `default_motor_pos` which represent `Stand` either. The bias comes
            # from motor backlash. and we count this slightly `bias` into config_motor.json `init_pos`.
            # NOTE: in CalibratePolicy, Integration controller is used, so the step_record_list[-1].motor_act
            # include the `bias`.
            motor_pos_bias = (
                # np.array(list(motor_angles_list[-1].values()), dtype=np.float32)
                step_record_list[-1].motor_act - policy.default_motor_pos
            )
            motor_pos_bias = np.where(  abs(motor_pos_bias) < 0.005,
                                        0.0,
                                        motor_pos_bias
                                        )
            logger.info(f'after calibrate policy, motor_pos_bias:{motor_pos_bias:.3f}')

            # motor_pos_delta[
            #     np.logical_and(motor_pos_delta > -0.005, motor_pos_delta < 0.005)
            # ] = 0.0

            with open(config_file, 'rt') as _f:
                motor_config = json.load(_f)

            # for motor_name, init_pos in zip(
            #     motor_names,  motor_pos_init + motor_pos_bias
            # ):
            for _x, _n in enumerate(robot.motor_name_ordering):
                # NOTE: can add a new key, or update existing key.
                motor_config[_n]["init_pos"] = robot.init_motor_angles[_n] + motor_pos_bias[_x]

            with open(config_file, 'wt') as _f:
                json.dump(motor_config, _f, indent=4)

            logger.info(f'updated joint_motor_mapping.json:  {json.dumps(motor_config, sort_keys=True, indent=4)}')

        else:
            raise FileNotFoundError(f"Could not find {config_file.resolve()}")

    # if isinstance(policy, PushCartPolicy):
    if type(policy).__name__ ==  'PushCartPolicy':
        video_path = log_dir / 'visual_obs.mp4'
        fps = int(1 / np.diff(policy.grasp_policy.camera_time_list).mean())
        logger.info(f"visual_obs fps: {fps}")
        video_clip = ImageSequenceClip(policy.grasp_policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if type(policy).__name__ == 'TeleopJoystickPolicy':
        policy_dict = {
            "hug": policy.hug_policy,
            "pick": policy.pick_policy,
            "grasp": policy.push_cart_policy.grasp_policy
            if hasattr(policy.push_cart_policy, "grasp_policy")
            else policy.teleop_policy,
        }
        for task_name, task_policy in policy_dict.items():
            if (
                not type(task_policy).__name__ == 'DPPolicy'
                or len(task_policy.camera_frame_list) == 0
            ):
                continue

            video_path = log_dir / f'{task_name}_visual_obs.mp4'
            fps = int(1 / np.diff(task_policy.camera_time_list).mean())
            logger.info(f"{task_name} visual_obs fps: {fps}")
            video_clip = ImageSequenceClip(task_policy.camera_frame_list, fps=fps)
            video_clip.write_videofile(video_path, codec="libx264", fps=fps)


# @profile()
def run_policy(*,
    robot: Robot, env: BaseEnv, policy: BasePolicy, vis_type: str, plot: bool
):
    """Executes a control policy on a robot within a simulation environment, logging data and optionally visualizing results.

    Args:
        robot (Robot): The robot instance to control.
        env (BaseEnv): sim or real. The simulation environment in which the robot operates.
        policy (BasePolicy): The control policy to execute.
        vis_type (str): The type of visualization to use ('view', 'render', etc.).
        plot (bool): Whether to plot the results after execution.
    """
    # header_name = snake2camel(env.env_name)

    # TODO: use multi-thread solution to record the following running data into log files.
    step_record_list: List[StepRecord] = []

    # loop_time_record_list: List[_StepTimeRecord] = []
    # obs_list: List[Obs] = []
    # control_inputs_list: List[Dict[str, float]] = []  # e.g., human operation.
    # motor_angles_list: List[Dict[str, float]] = []
    # motor_angles_list: List[npt.NDArray[np.float32]] = []


    # TODO: BasePolicy.n_steps_total defaults to `inf`, and only sysIDPolicy override it to len(time_seq).
    # n_steps_total = (
    #     float("inf")
    #     if "real" in env.name and "fixed" not in policy.name
    #     else policy.n_steps_total
    # )

    # TODO: for tqdm,  if total is float('inf'), Infinite iterations,
    #  behave same as `total-unknown`: can not show progress bar.
    # not use tqdm for n_steps_total is inf?
    # p_bar = tqdm(total=policy.n_steps_total, desc="Running the policy",
    #              colour='CYAN', unit='step', unit_scale=True)
    run_start_time = timelib.time()
    _step_count:int = 0
    time_until_next_step = 0.0
    # update tqdm every 1 sec.
    p_bar_steps:int = max(1, int(1 / policy.control_dt_sec))

    # for sysID only.
    # _cur_ep_idx :int = -1
    motor_kp_setter: _MotorKpSetter | None = _MotorKpSetter() \
        if type(policy).__name__ == 'SysIDPolicy' else None

    # TODO: for tqdm,  if total is float('inf'), Infinite iterations,
    #  behave same as `total-unknown`: can not show progress bar.
    # not use tqdm for n_steps_total is inf?
    with  tqdm(total=policy.n_steps_total, desc="Running the policy",
                 colour='CYAN', unit='step', unit_scale=True) as p_bar:
        try:
            while _step_count < policy.n_steps_total:
                _record = StepRecord()
                _record.time_pnt.step_start = timelib.time()

                # Get the latest state from the queue
                obs = env.get_observation(1)
                # change to epoch time.
                obs.time -= run_start_time

                if "real" not in env.env_name and vis_type != "view":
                    obs.time += time_until_next_step

                _record.time_pnt.recv_obs = timelib.time()

                # for sysID policy to change motor kp if kp changed.
                if motor_kp_setter is not None:
                    motor_kp_setter.set_kp(policy=policy, env=env,step_count=_step_count,obs_time=obs.time)

                # for TeleopLeaderPolicy and RecordPolicy to toggle motor torque.
                # # need to enable and disable motors according to logging state
                _toggle_motor(policy, env)

                control_inputs, motor_target_arr = policy.step(obs, "real" in env.env_name)
                _record.time_pnt.inference = timelib.time()

                assert len(motor_target_arr) == len(robot.motor_name_ordering)
                # motor_angle_dict: Dict[str, float] = OrderedDict(zip(robot.motor_name_ordering, motor_target_arr))

                # every 6 seconds.
                if _step_count % 300 == 1:
                    # NOTE: set/get value should be normalized by feite_controller.init_pos
                    logger.info(f'prev act:{step_record_list[-1].motor_act}, {obs.motor_pos=:}, {motor_target_arr=:}')

                # env.set_motor_target(motor_angle_dict)
                env.set_motor_target(motor_target_arr)
                _record.time_pnt.set_action = timelib.time()

                env.step()

                _record.time_pnt.sim_step = timelib.time()

                _record.obs=deepcopy(obs)
                _record.ctrl_input = deepcopy(control_inputs)
                _record.motor_act = deepcopy(motor_target_arr)


                # obs_list.append(obs)
                # control_inputs_list.append(control_inputs)
                # motor_angles_list.append(motor_target_arr)

                _step_count += 1

                # update tqdm every 1 sec (time measured in policy.control_dt).
                if _step_count % p_bar_steps == 0:
                    p_bar.update(p_bar_steps)

                _record.time_pnt.step_end = timelib.time()

                # loop_time_record_list.append( time_record)

                time_until_next_step = (run_start_time +
                                        policy.control_dt_sec * _step_count
                                        - _record.time_pnt.step_end)

                step_record_list.append(_record)

                logger.debug(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")

                if time_until_next_step < 0:
                    logger.warning(f'time_until_next_step < 0 : {time_until_next_step}')

                if ("real" in env.env_name or vis_type == "view") and time_until_next_step > 0:
                    logger.debug(f'+++++ sleep for {time_until_next_step * 1000:.2f} ms')
                    timelib.sleep(time_until_next_step)

        except KeyboardInterrupt:
            # only catch Keyboard Interrupt as normal exit from while loop,
            # and save running logs in and after `finally` block.
            logger.warning("KeyboardInterrupt received. exit while loop, and save running logs." )

        except Exception as err:
            # other exceptions, like IOError, re-raise the exception to outer `try.. ex...fi..`.
            # without saving running logs.
            # NOTE: the `finally` block will be executed before re-raise to outer `try` block.
            logger.error(f'Unexpected error occurred: {err=:}, {type(err)=:}. re-raise to outer handler.')
            raise

        finally:
            # p_bar.close()

            logger.info(f'exit from run while loop, final step_count: {_step_count},'
                        f' step record count: {len(step_record_list)}')

            # TODO: save recording file every n steps n seconds. ... not at the end of while loop.....
            # exp_name = f"{robot.name}_{policy.name}_{env.env_name}"
            # exp_folder = Path('run_policy_log') / f'{exp_name}_{cur_time}'
            # 'run_policy_log/{robot_name}_{policy_name}_{env_name}_{cur_time}'
            cur_time = timelib.strftime("%Y%m%d_%H%M%S")
            exp_folder = Path(RUN_POLICY_LOG_FOLDER_FMT.format(robot_name=robot.name,
                                                          policy_name=policy.name,
                                                          env_name=env.env_name,
                                                          cur_time=cur_time) )

            exp_folder.mkdir(parents=True, exist_ok=True)
            # os.makedirs(exp_folder, exist_ok=True)

            # TODO: save recording file every n steps n seconds. ... not at the end of while loop.....
            if vis_type == "render" and isinstance(env, MuJoCoSim):   #hasattr(env, "save_recording"):
                # assert isinstance(env, MuJoCoSim)
                env.save_recording(exp_folder, policy.control_dt_sec, 2)

            # Using context mgr to close env, not use close() standalone.
            # close() also set torque off for all connected motors.
            # env.close()

            # ----  at end of `finally` execution, if there is un-handled Exp, will raise to outer `try` block; else,
            # execution continues the following code.

    # ---- save logs only when: 1. finish while loop; 2. KeyboardInterrupt. ----

    # TODO: write log data every n steps..n seconds.. not at the end of while loop.....

    # log_data_dict: Dict[str, Any] = {
    #     "obs_list": obs_list,
    #     "control_inputs_list": control_inputs_list,
    #     "motor_angles_list": motor_angles_list,
    # }

    _save_run_log(step_record_list, exp_folder / RUN_STEP_RECORD_PICKLE_FILE)
    _save_policy_log(policy=policy,robot=robot,
                     log_dir=exp_folder, # / policy.name,
                     step_record_list=step_record_list)

    dump_profiling_data(exp_folder / 'profile_output.lprof')
    if plot:
        logger.info("Plot policy run logg->")
        _plot_run_log(
            robot,
            step_record_list,
            exp_folder / 'plot'
        )
        # plot_results(
        #     robot,
        #     loop_time_record_list,
        #     obs_list,
        #     control_inputs_list,
        #     motor_angles_list,
        #     exp_folder,
        # )


@contextmanager
def _build_env(args: argparse.Namespace, robot: Robot)->Generator[Optional[BaseEnv],None, None]:
    # env: BaseEnv | None = None

    if args.env == "mujoco":
        env = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.policy)
        # init_motor_pos = env.get_observation(1).motor_pos

    elif args.env == "real":
        # TODO: `fixed` in input args, no use for ReadWorld?
        env = RealWorld(robot)
        # init_motor_pos = env.get_observation(1).motor_pos
    else:
        raise ValueError(f"Unknown simulator:{args.env}")

    try:
        yield env   # exception entering here.
    except IOError as err:
        # NOTE: even we call exit(0) here, the `finally` segment will be guaranteed to be executed before program exit.
        # exit(0)
        logger.error(f'IO Error occurred, please check the motor connection. {err=:}')
        raise
    except KeyboardInterrupt:
        logger.error(f'keyboard interrupt. exiting...')
    except Exception as err:
        logger.error(f'Unexpected error occurred: {err=:}, {type(err)=:} ')
        # NOTE: even we call exit(0) here, the `finally` segment will be guaranteed to be executed before program exit.
        # exit(0)
        raise
    finally:
        # release resources.
        logger.warning(f'---> closing env: {env.env_name}')
        if env is not None:
            env.close()
            # pause 1 second to wait for the serial port close complete, before exiting from program.
            time.sleep(1.)


def _build_policy(args:argparse.Namespace, robot:Robot, init_motor_pos:npt.NDArray[np.float32] )->BasePolicy:
    # policy: BasePolicy | None = None

    # TODO: confusing.  we can separate them into two fields:  --policy xxx  --fixed true/false.
    # `fixed` meas robot with a fixed base , e.g. fixed by a bench clamp.
    # NOTE: all policy name can be added with a "_fixed" suffix during input args.
    PolicyCls = _get_policy_class_v2(args.policy.replace("_fixed", ""))
    policy_cls_base_name = {_base.__name__ for _base in PolicyCls.__bases__}
    logger.info(f'get policy class: {PolicyCls.__name__}, base class: {policy_cls_base_name}')

    if "replay" in args.policy:
        policy = PolicyCls(args.policy, robot, init_motor_pos, args.run_name)

    elif "teleop_leader" in args.policy:
        assert args.robot == "toddlerbot_arms", (
            "The teleop leader policy is only for the arms"
        )
        assert args.env == "real", (
            "The env needs to be the real world for the teleop leader policy"
        )
        for motor_name in robot.motor_name_ordering:
            for gain_name in ["kp_real", "kd_real", "kff1_real", "kff2_real"]:
                robot.config["joints"][motor_name][gain_name] = 0.0

        policy = PolicyCls(
            args.policy, robot, init_motor_pos, ip=args.ip, task=args.task
        )  # type: ignore

    elif "teleop_follower" in args.policy:
        # Run the command
        if len(args.ip) > 0:
            sync_time(args.ip)

        policy = PolicyCls(
            args.policy, robot, init_motor_pos, ip=args.ip, task=args.task
        )  # type: ignore

    elif "teleop_joystick" in args.policy:
        if len(args.ip) > 0:
            sync_time(args.ip)

        policy = PolicyCls(  # type: ignore
            args.policy, robot, init_motor_pos, ip=args.ip, run_name=args.run_name
        )

    elif "push_cart" in args.policy:
        policy = PolicyCls(args.policy, robot, init_motor_pos, args.ckpt)

    # elif issubclass(PolicyCls, MJXPolicy):
    elif 'MJXPolicy' in policy_cls_base_name:
        fixed_command = None
        if len(args.command) > 0:
            fixed_command = np.array(args.command.split(" "), dtype=np.float32)

        policy = PolicyCls(
            args.policy, robot, init_motor_pos, args.ckpt, fixed_command=fixed_command
        )

    # elif issubclass(PolicyCls, DPPolicy):
    elif 'DPPolicy' in policy_cls_base_name:
        policy = PolicyCls(
            args.policy, robot, init_motor_pos, args.ckpt, task=args.task
        )

    # elif issubclass(PolicyCls, BalancePDPolicy):
    elif 'BalancePDPolicy' in policy_cls_base_name:
        # Run the command
        if len(args.ip) > 0:
            sync_time(args.ip)

        fixed_command = None
        if len(args.command) > 0:
            fixed_command = np.array(args.command.split(" "), dtype=np.float32)

        policy = PolicyCls(
            args.policy, robot, init_motor_pos, ip=args.ip, fixed_command=fixed_command
        )
    elif "talk" in args.policy:
        policy = PolicyCls(args.policy, robot, init_motor_pos, ip=args.ip)  # type:ignore
    else:
        policy = PolicyCls(args.policy, robot, init_motor_pos)

    return policy


def _main(args:argparse.Namespace):
    """Executes a policy for a specified robot and simulator configuration.

    This function parses command-line arguments to configure and run a policy for a robot. It supports different robots, simulators, visualization types, and tasks. The function initializes the appropriate simulation environment and policy based on the provided arguments and executes the policy.

    Args:
        args (list, optional): List of command-line arguments. If None, defaults to sys.argv.

    Raises:
        ValueError: If an unknown simulator is specified.
        AssertionError: If the teleop leader policy is used with an unsupported robot or simulator.
    """

    robot = Robot(args.robot)

    # t1 = timelib.time()

    # env: BaseEnv = _build_env(args, robot)
    with _build_env(args, robot) as env:
        logger.info(f'create env: {env.env_name}')

        # t2 = timelib.time()

        init_motor_pos: npt.NDArray[np.float32] = env.get_observation(1).motor_pos
        logger.info(f'read init motor pos: {init_motor_pos}')

        policy:BasePolicy = _build_policy(args, robot, init_motor_pos)

        # t3 = timelib.time()

        # print(f"Time taken to initialize env: {t2 - t1:.2f} s")
        # print(f"Time taken to initialize policy: {t3 - t2:.2f} s")

        run_policy(robot=robot,env=env,policy=policy,vis_type=args.vis, plot=args.plot)


def _args_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='toddler: run a policy.')
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        # "--sim",
        "--env",
        type=str,
        default="mujoco",
        help="The name of the environment to use.",
        choices=["mujoco", "real"],
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="render",
        help="The visualization type.",
        choices=["render", "view", "none"],
    )

    # TODO: confusing.  we can separate them into two fields:  --policy xxx  --fixed true/false.
    parser.add_argument(
        "--policy",
        type=str,
        default="stand",
        help="The name of the task.",
        choices=_get_policy_names_v2(),
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="The policy checkpoint to load for RL policies.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="The policy checkpoint to load for RL policies.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        dest='run_name',
        help="The policy run to replay.",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="",
        help="The ip address of the follower.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hug",
        choices=["hug", "pick", "grasp"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Skip the plot functions.",
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
                   module_logger_config={'toddler.policies': logging.INFO,
                                         'main': logging.INFO})
    # use root logger for __main__.
    logger = logging.getLogger('root')
    logger.info('parsed args --->\n{}'.format('\n'.join(
        f'{arg_name}={arg_value}' for arg_name, arg_value in
        sorted(_parsed_args.__dict__.items(), key=lambda k_v_pair: k_v_pair[0]))))

    _main(_parsed_args)

