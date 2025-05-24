import argparse
# import bisect
import importlib
import json
import os
import pickle
import pkgutil
import time
import time as timelib
from typing import Any, Dict, List,Optional, Generator
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from moviepy import ImageSequenceClip
from tqdm import tqdm
import logging


from . import *
# from toddlerbot.policies import BasePolicy, get_policy_class, get_policy_names
# from toddlerbot.policies.balance_pd import BalancePDPolicy
# from toddlerbot.policies.calibrate import CalibratePolicy
# from toddlerbot.policies.dp_policy import DPPolicy
# from toddlerbot.policies.mjx_policy import MJXPolicy
# from toddlerbot.policies.push_cart import PushCartPolicy
# from toddlerbot.policies.record import RecordPolicy
# from toddlerbot.policies.replay import ReplayPolicy
# from toddlerbot.policies.sysID import SysIDPolicy
# from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
# from toddlerbot.policies.teleop_joystick import TeleopJoystickPolicy
# from toddlerbot.policies.teleop_leader import TeleopLeaderPolicy

from toddlerbot.sim import BaseEnv, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot

# from toddlerbot.utils.comm_utils import sync_time
# from toddlerbot.utils.misc_utils import dump_profiling_data, log, snake2camel
# from ..utils.config_logging import config_logging

from ..utils import sync_time, dump_profiling_data, snake2camel, config_logging

from ..visualization import *

# from toddlerbot.visualization.vis_plot import (
#     plot_joint_tracking,
#     plot_joint_tracking_frequency,
#     plot_joint_tracking_single,
#     plot_line_graph,
#     plot_loop_time,
#     plot_motor_vel_tor_mapping,
#     # plot_path_tracking,
# )

# from toddlerbot.utils.misc_utils import profile
from ._module_logger import logger

def dynamic_import_policies(policy_package: str):
    """Dynamically imports all modules within a specified package.

    This function attempts to import each module found in the given package directory. If a module cannot be imported, a log message is generated.

    Args:
        policy_package (str): The name of the package containing the modules to be imported.
    """
    package = importlib.import_module(policy_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{policy_package}.{module_name}"
        try:
            importlib.import_module(full_module_name)
        except Exception:
            log(f"Could not import {full_module_name}", header="Dynamic Import")


# Call this to import all policies dynamically
dynamic_import_policies("toddlerbot.policies")


def plot_results(
    robot: Robot,
    loop_time_list: List[List[float]],
    obs_list: List[Obs],
    control_inputs_list: List[Dict[str, float]],
    motor_angles_list: List[Dict[str, float]],
    exp_folder_path: str,
):
    """Generates and saves various plots to visualize the performance and behavior of a robot during an experiment.

    Args:
        robot (Robot): The robot object containing information about the robot's configuration and state.
        loop_time_list (List[List[float]]): A list of lists containing timing information for each loop iteration.
        obs_list (List[Obs]): A list of observations recorded during the experiment.
        control_inputs_list (List[Dict[str, float]]): A list of dictionaries containing control inputs applied to the robot.
        motor_angles_list (List[Dict[str, float]]): A list of dictionaries containing motor angles recorded during the experiment.
        exp_folder_path (str): The path to the folder where the plots will be saved.
    """
    loop_time_dict: Dict[str, List[float]] = {
        "obs_time": [],
        "inference": [],
        "set_action": [],
        "sim_step": [],
        "log_time": [],
        # "total_time": [],
    }
    for i, loop_time in enumerate(loop_time_list):
        (
            step_start,
            obs_time,
            inference_time,
            set_action_time,
            sim_step_time,
            step_end,
        ) = loop_time
        loop_time_dict["obs_time"].append((obs_time - step_start) * 1000)
        loop_time_dict["inference"].append((inference_time - obs_time) * 1000)
        loop_time_dict["set_action"].append(
            (set_action_time - inference_time) * 1000
        )
        loop_time_dict["sim_step"].append((sim_step_time - set_action_time) * 1000)
        loop_time_dict["log_time"].append((step_end - sim_step_time) * 1000)
        # loop_time_dict["total_time"].append((step_end - step_start) * 1000)

    time_obs_list: List[float] = []
    # lin_vel_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    pos_obs_list: List[npt.NDArray[np.float32]] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    tor_obs_total_list: List[float] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    motor_pos_dict: Dict[str, List[float]] = {}
    motor_vel_dict: Dict[str, List[float]] = {}
    motor_tor_dict: Dict[str, List[float]] = {}
    for i, obs in enumerate(obs_list):
        time_obs_list.append(obs.time)
        # lin_vel_obs_list.append(obs.lin_vel)
        ang_vel_obs_list.append(obs.ang_vel)
        pos_obs_list.append(obs.pos)
        euler_obs_list.append(obs.euler)
        tor_obs_total_list.append(sum(obs.motor_tor))

        for j, motor_name in enumerate(robot.motor_name_ordering):
            if motor_name not in time_seq_dict:
                time_seq_ref_dict[motor_name] = []
                time_seq_dict[motor_name] = []
                motor_pos_dict[motor_name] = []
                motor_vel_dict[motor_name] = []
                motor_tor_dict[motor_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[motor_name].append(float(obs.time))
            time_seq_ref_dict[motor_name].append(float(obs.time))
            # time_seq_ref_dict[motor_name].append(i * policy.control_dt)
            motor_pos_dict[motor_name].append(obs.motor_pos[j])
            motor_vel_dict[motor_name].append(obs.motor_vel[j])
            motor_tor_dict[motor_name].append(obs.motor_tor[j])

    action_dict: Dict[str, List[float]] = {}
    joint_pos_ref_dict: Dict[str, List[float]] = {}
    for motor_angles in motor_angles_list:
        for motor_name, motor_angle in motor_angles.items():
            if motor_name not in action_dict:
                action_dict[motor_name] = []
            action_dict[motor_name].append(motor_angle)

        joint_angle_ref = robot.motor_to_active_joint_angles(motor_angles)
        for joint_name, joint_angle in joint_angle_ref.items():
            if joint_name not in joint_pos_ref_dict:
                joint_pos_ref_dict[joint_name] = []
            joint_pos_ref_dict[joint_name].append(joint_angle)

    control_inputs_dict: Dict[str, List[float]] = {}
    for control_inputs in control_inputs_list:
        for control_name, control_input in control_inputs.items():
            if control_name not in control_inputs_dict:
                control_inputs_dict[control_name] = []
            control_inputs_dict[control_name].append(control_input)

    plt.switch_backend("Agg")

    plot_loop_time(loop_time_dict, exp_folder_path)

    if "sysID" in robot.name:
        plot_motor_vel_tor_mapping(
            motor_vel_dict["joint_0"],
            motor_tor_dict["joint_0"],
            save_path=exp_folder_path,
        )

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
    #         save_path=exp_folder_path,
    #         file_name="com_tracking",
    #     )()

    plot_line_graph(
        tor_obs_total_list,
        time_obs_list,
        legend_labels=["Torque (Nm) or Current (mA)"],
        title="Total Torque or Current  Over Time",
        x_label="Time (s)",
        y_label="Torque (Nm) or Current (mA)",
        save_config=True,
        save_path=exp_folder_path,
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
        save_path=exp_folder_path,
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
        save_path=exp_folder_path,
        file_name="euler_tracking",
    )()
    # if len(control_inputs_dict) > 0:
    #     plot_path_tracking(
    #         time_obs_list,
    #         pos_obs_list,
    #         euler_obs_list,
    #         control_inputs_dict,
    #         save_path=exp_folder_path,
    #     )
    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        robot.joint_cfg_limits,
        save_path=exp_folder_path,
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_tor_dict,
        save_path=exp_folder_path,
        y_label="Torque (Nm) or Current (mA)",
        file_name="motor_tor_tracking",
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_vel_dict,
        save_path=exp_folder_path,
    )
    plot_joint_tracking_frequency(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        save_path=exp_folder_path,
    )


@dataclass(init=True)
class _StepTimeRecord:
    step_start:float = float('inf')
    recv_obs: float = float('inf')
    inference: float = float('inf')
    set_action: float = float('inf')
    sim_step: float = float('inf')
    step_end: float = float('inf')


@dataclass(init=True)
class _MotorKpSetter:
    _cur_ep_idx:int = -1

    def set_kp(self, *,
                     policy:BasePolicy, env:BaseEnv, step_count:int, obs_time: float):
        assert isinstance(policy, SysIDPolicy)
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
            assert obs_time <= policy.episode_motor_kp[0].ep_end_time_pnt
            self._cur_ep_idx = 0

            env.set_motor_kps(policy.episode_motor_kp[self._cur_ep_idx].motor_kp)

        # TODO: if len(ep) == 1?
        elif self._cur_ep_idx == len(policy.episode_motor_kp) - 1:
            # already last episode, not set.
            pass

        elif obs_time > policy.episode_motor_kp[self._cur_ep_idx].ep_end_time_pnt:
            # we do not allow obs skip an episode.
            assert obs_time <= policy.episode_motor_kp[self._cur_ep_idx + 1].ep_end_time_pnt
            self._cur_ep_idx += 1

            # TODO: if all kp are zero, not set ,keep kp value set previously? but episode_motor_kp
            # will be recorded into log_data_dict....
            # not update if all zero. but motor is already set previous kp value, no change?
            # if np.any(tuple(motor.values())):
            #     env.set_motor_kps(motor_kps_updated)
            #     prev_ep_idx = ep_idx

            env.set_motor_kps(policy.episode_motor_kp[self._cur_ep_idx].motor_kp)
        else:
            # kp no change, not set.
            pass



def _toggle_motor(policy:BasePolicy, env:BaseEnv):
    # need to enable and disable motors according to logging state
    if isinstance(policy, TeleopLeaderPolicy) and policy.toggle_motor:
        assert isinstance(env, RealWorld)
        if policy.is_running:
            # disable all motors when logging
            env.actuator_controller.disable_motors()
        else:
            # enable all motors when not logging
            env.actuator_controller.enable_motors()

        policy.toggle_motor = False

    elif isinstance(policy, RecordPolicy) and policy.toggle_motor:
        assert isinstance(env, RealWorld)
        env.actuator_controller.disable_motors(policy.disable_motor_indices)
        policy.toggle_motor = False



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
    header_name = snake2camel(env.env_name)

    loop_time_record_list: List[_StepTimeRecord] = []
    obs_list: List[Obs] = []
    control_inputs_list: List[Dict[str, float]] = []  # e.g., human operation.
    motor_angles_list: List[Dict[str, float]] = []

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
    p_bar_steps:int = int(1 / policy.control_dt)

    # for sysID only.
    # _cur_ep_idx :int = -1
    motor_kp_setter: _MotorKpSetter = _MotorKpSetter()

    # TODO: for tqdm,  if total is float('inf'), Infinite iterations,
    #  behave same as `total-unknown`: can not show progress bar.
    # not use tqdm for n_steps_total is inf?
    with  tqdm(total=policy.n_steps_total, desc="Running the policy",
                 colour='CYAN', unit='step', unit_scale=True) as p_bar:
            try:
                while _step_count < policy.n_steps_total:
                    time_record = _StepTimeRecord()
                    time_record.step_start = timelib.time()

                    # Get the latest state from the queue
                    obs = env.get_observation(1)
                    # change to epoch time.
                    obs.time -= run_start_time

                    if "real" not in env.env_name and vis_type != "view":
                        obs.time += time_until_next_step

                    time_record.recv_obs = timelib.time()

                    # for sysID policy to change motor kp if kp changed.
                    motor_kp_setter.set_kp(policy=policy, env=env,step_count=_step_count,obs_time=obs.time)

                    # for TeleopLeaderPolicy and RecordPolicy to toggle motor torque.
                    # # need to enable and disable motors according to logging state
                    _toggle_motor(policy, env)

                    control_inputs, motor_target_arr = policy.step(obs, "real" in env.env_name)
                    time_record.inference = timelib.time()

                    assert len(motor_target_arr) == len(robot.motor_name_ordering)
                    motor_angle_dict: Dict[str, float] = OrderedDict(zip(robot.motor_name_ordering, motor_target_arr))

                    # every 6 seconds.
                    if _step_count % 300 == 0:
                        logger.info(f'{motor_angle_dict=:}')

                    # for _name, _angle in zip(robot.motor_name_ordering, motor_target):
                    #     motor_angles[_name] = _angle

                    env.set_motor_target(motor_angle_dict)
                    time_record.set_action = timelib.time()

                    env.step()
                    time_record.sim_step = timelib.time()

                    obs_list.append(obs)
                    control_inputs_list.append(control_inputs)
                    motor_angles_list.append(motor_angles)

                    _step_count += 1

                    # update tqdm every 1 sec (time measured in policy.control_dt).
                    if _step_count % p_bar_steps == 0:
                        p_bar.update(p_bar_steps)

                    time_record.step_end = timelib.time()

                    loop_time_record_list.append( time_record )
                        # [
                        #     step_start,
                        #     cur_loop_start_time,
                        #     inference,
                        #     set_action,
                        #     sim_step,
                        #     step_end,
                        # ]
                    # )

                    time_until_next_step = (run_start_time +
                                            policy.control_dt * _step_count
                                            - time_record.step_end)
                    logger.debug(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")

                    assert time_until_next_step always < 0????

                    if ("real" in env.env_name or vis_type == "view") and time_until_next_step > 0:
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

            # TODO: save recording file every n steps n seconds. ... not at the end of while loop.....
            exp_name = f"{robot.name}_{policy.name}_{env.env_name}"
            time_str = timelib.strftime("%Y%m%d_%H%M%S")
            exp_folder_path = f"results/{exp_name}_{time_str}"

            os.makedirs(exp_folder_path, exist_ok=True)

            # TODO: save recording file every n steps n seconds. ... not at the end of while loop.....
            if vis_type == "render" and isinstance(env, MuJoCoSim):   #hasattr(env, "save_recording"):
                # assert isinstance(env, MuJoCoSim)
                env.save_recording(exp_folder_path, policy.control_dt, 2)

            # Using context mgr to close env, not use close() standalone.
            # close() also set torque off for all connected motors.
            # env.close()

            # ----  at end of `finally` execution, if there is un-handled Exp, will raise to outer `try` block; else,
            # execution continues the following code.

    # ---- save logs only when: 1. finish while loop; 2. KeyboardInterrupt. ----

    # TODO: write log data every n steps..n seconds.. not at the end of while loop.....

    log_data_dict: Dict[str, Any] = {
        "obs_list": obs_list,
        "control_inputs_list": control_inputs_list,
        "motor_angles_list": motor_angles_list,
    }

    if isinstance(policy, SysIDPolicy):
        log_data_dict["episode_motor_kp"] = policy.episode_motor_kp

    log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
    with open(log_data_path, "wb") as f:
        pickle.dump(log_data_dict, f)

    prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
    dump_profiling_data(prof_path)

    if isinstance(policy, TeleopFollowerPDPolicy):
        policy.dataset_logger.move_files_to_exp_folder(exp_folder_path)

    if isinstance(policy, DPPolicy) and len(policy.camera_frame_list) > 0:
        fps = int(1 / np.diff(policy.camera_time_list).mean())
        log(f"visual_obs fps: {fps}", header=header_name)
        video_path = os.path.join(exp_folder_path, "visual_obs.mp4")
        video_clip = ImageSequenceClip(policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if isinstance(policy, ReplayPolicy):
        with open(os.path.join(exp_folder_path, "keyframes.pkl"), "wb") as f:
            pickle.dump(policy.keyframes, f)

    if isinstance(policy, CalibratePolicy):
        motor_config_path = os.path.join(robot.root_path, "config_motors.json")
        if os.path.exists(motor_config_path):
            motor_names = robot.get_joint_config_attrs("is_passive", False)
            motor_pos_init = np.array(
                robot.get_joint_config_attrs("is_passive", False, "init_pos")
            )
            motor_pos_delta = (
                np.array(list(motor_angles_list[-1].values()), dtype=np.float32)
                - policy.default_motor_pos
            )
            motor_pos_delta[
                np.logical_and(motor_pos_delta > -0.005, motor_pos_delta < 0.005)
            ] = 0.0

            with open(motor_config_path, "r") as f:
                motor_config = json.load(f)

            for motor_name, init_pos in zip(
                motor_names, motor_pos_init + motor_pos_delta
            ):
                motor_config[motor_name]["init_pos"] = float(init_pos)

            with open(motor_config_path, "w") as f:
                json.dump(motor_config, f, indent=4)
        else:
            raise FileNotFoundError(f"Could not find {motor_config_path}")

    if isinstance(policy, PushCartPolicy):
        video_path = os.path.join(exp_folder_path, "visual_obs.mp4")
        fps = int(1 / np.diff(policy.grasp_policy.camera_time_list).mean())
        log(f"visual_obs fps: {fps}", header=header_name)
        video_clip = ImageSequenceClip(policy.grasp_policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if isinstance(policy, TeleopJoystickPolicy):
        policy_dict = {
            "hug": policy.hug_policy,
            "pick": policy.pick_policy,
            "grasp": policy.push_cart_policy.grasp_policy
            if hasattr(policy.push_cart_policy, "grasp_policy")
            else policy.teleop_policy,
        }
        for task_name, task_policy in policy_dict.items():
            if (
                not isinstance(task_policy, DPPolicy)
                or len(task_policy.camera_frame_list) == 0
            ):
                continue

            video_path = os.path.join(exp_folder_path, f"{task_name}_visual_obs.mp4")
            fps = int(1 / np.diff(task_policy.camera_time_list).mean())
            log(f"{task_name} visual_obs fps: {fps}", header=header_name)
            video_clip = ImageSequenceClip(task_policy.camera_frame_list, fps=fps)
            video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if plot:
        log("Visualizing...", header=header_name)
        plot_results(
            robot,
            loop_time_record_list,
            obs_list,
            control_inputs_list,
            motor_angles_list,
            exp_folder_path,
        )


@contextmanager
def _build_env(args: argparse.Namespace, robot: Robot)->Generator[Optional[BaseEnv]]:
    env: BaseEnv | None = None

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

    policy: BasePolicy | None = None

    # TODO: confusing.  we can separate them into two fields:  --policy xxx  --fixed true/false.
    # `fixed` meas robot with a fixed base , e.g. fixed by a bench clamp.
    # NOTE: all policy name can be added with a "_fixed" suffix during input args.
    PolicyCls = get_policy_class(args.policy.replace("_fixed", ""))
    logger.info(f'get policy class: {PolicyCls.__name__}')

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

    elif issubclass(PolicyCls, MJXPolicy):
        fixed_command = None
        if len(args.command) > 0:
            fixed_command = np.array(args.command.split(" "), dtype=np.float32)

        policy = PolicyCls(
            args.policy, robot, init_motor_pos, args.ckpt, fixed_command=fixed_command
        )

    elif issubclass(PolicyCls, DPPolicy):
        policy = PolicyCls(
            args.policy, robot, init_motor_pos, args.ckpt, task=args.task
        )

    elif issubclass(PolicyCls, BalancePDPolicy):
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
        logger.info(f'create env: {env.__name__}')

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
        choices=get_policy_names(),
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
                   module_logger_config={'policies': logging.INFO,
                                         'main': logging.INFO})
    # use root logger for __main__.
    logger = logging.getLogger('root')
    logger.info('parsed args --->\n{}'.format('\n'.join(
        f'{arg_name}={arg_value}' for arg_name, arg_value in
        sorted(_parsed_args.__dict__.items(), key=lambda k_v_pair: k_v_pair[0]))))

    _main(_parsed_args)

