
import functools
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from datetime import datetime
import mediapy as media

from mujoco_playground import wrapper, registry
from mujoco_playground._src.gait import draw_joystick_command

from kbot.locomotion.kbot_both_leg.joystick.try_g1_joystick_env import TryG1Joystick

NUM_EPISODE = 1
ROLL_OUT_CMD=dict(
    x_vel = 0.156,
    y_vel = 0.0,
    yaw_vel = 0.1 * jnp.pi  #  0.87 * jp.pi,
)
INCREASE_CMD=True
# EVAL_EPISODE_LEN = 1_400


def get_env_wrapper(env_name: str) -> Callable:
    return wrapper.wrap_for_brax_training

def get_domain_randomizer(env_name:str)->Callable:
    return registry.get_domain_randomizer(env_name)

def eval_env(env_name:str):
    del env_name
    # Enable perturbation in the eval env.
    return TryG1Joystick('flat_terrain')

def train_env(env_name:str):
    del env_name
    return TryG1Joystick('flat_terrain')


def rollout(*,
            env,
            jit_reset: Callable,
            jit_infer_fn: Callable,
            jit_step: Callable,
            rng: jax.Array,
            record_step_state: bool
            ):
    env_cfg = env._config
    phase_dt = 2 * jnp.pi * env.dt * 1.5
    phase = jnp.array([0, jnp.pi])
    command = jnp.array([ROLL_OUT_CMD['x_vel'],
                        ROLL_OUT_CMD['y_vel'],
                        ROLL_OUT_CMD['yaw_vel'] ])

    print(f"Initial command : {command}")

    # NOTE: put the data to be saved onto host-CPU to make the `pickle`
    # loading faster during plot stage.
    ro_data = dict(
        states=[],
        modify_scene_fns=[],
        command=[]
    )

    # TODO: NOTE: only state.data is trustable, state.info is
    # keeping modified during training/eval invokation of step().
    def _collect_ro_data(state):
        # TODO: NOTE: only state.data is trustable, state.info is
        # keeping modified during training/eval invokation of step().
        if record_step_state:
            ro_data['states'].append(state)

        # move from device('gpu') to host cpu.(as a copy), cause state.info contents will
        # keeping be changed during train/eval step().
        # jp.zeros/jp.maximum etc. will place the array on device('gpu') automatically.
        # device_get() return a numpy.array on cpu.
        ro_data['command'].append(jax.device_get(state.info["command"]))

        # in global frame.
        # np.array will move jp.array from device to host cpu implicitly.
        # same effect as device_get.
        # cmd_arrow_global_xyz = np.array(state.data.xpos[env._torso_body_id])

        # kenneth: pelvis body id is 1.
        cmd_arrow_global_xyz = np.array(state.data.xpos[1])

        # arrow placed 0. over torsor
        cmd_arrow_global_xyz += np.array([0., 0., 0.])

        # xmat [14,3,3]: transform matrix, 3-by-3 per body.
        # x_axis = jax.device_get(state.data.xmat[env._torso_body_id, 0])

        # kenneth: pelvis body id is 1.
        x_axis = jax.device_get(state.data.xmat[1, 0])
        cmd_arrow_global_yaw = -np.arctan2(x_axis[1], x_axis[0])

        ro_data['modify_scene_fns'].append(
            functools.partial(
                draw_joystick_command,
                rgba=[0.7, 0.1, 0.1, 0.5],
                cmd=ro_data['command'][-1],
                xyz=cmd_arrow_global_xyz,
                theta=cmd_arrow_global_yaw,
                radius=0.01,
                scl=np.linalg.norm(ro_data['command'][-1]),
            )
        )

    for _ep in range(NUM_EPISODE):
        print(f"episode {_ep} --->")
        rng, key = jax.random.split(rng)

        state = jit_reset(key)
        state.info["phase_dt"] = phase_dt
        state.info["phase"] = phase
        # overwrite info['command'] set in reset().
        state.info["command"] = command

        ep_step_times = [datetime.now(),]

        for i in range(env_cfg.episode_length):
            # Increase the forward velocity by 0.25 m/s every 200 steps.
            if INCREASE_CMD and i % 200 == 0:
                command = command.at[0].add(0.25)
                command = jnp.clip(command, -0.9, 0.9)
                print(f"Setting command to {command}")

            act_key, rng = jax.random.split(rng)
            #TODO: act_rng is no use when deterministic==True.
            ctrl, _ = jit_infer_fn(state.obs, act_key)
            state = jit_step(state, ctrl)
            # overwrite info['command'] set in step() through sample_command().
            state.info["command"] = command   # as obs member.
            ep_step_times.append(datetime.now())
            # TODO: NOTE: only state.data is trustable, state.info is
            # keeping modified during training/eval invoke of step().
            _collect_ro_data(state)

            if state.done:
                print(f"episode done after {ep_step_times[-1] - ep_step_times[1]}")
                break

        print(f"episode: {_ep} ---> rollout time to jit: {ep_step_times[1] - ep_step_times[0]}\n"
              f"rollout time to eval: {ep_step_times[-1] - ep_step_times[1]}")

    return ro_data


def render_to_video(env, ro_data, video_file_path):
    render_every = 1
    fps = 1.0 / env.dt / render_every
    print(f"render video fps: {fps}")
    traj = ro_data['states'][::render_every]   # only use state.data.
    mod_fns = ro_data['modify_scene_fns'][::render_every]

    scene_option = mujoco.MjvOption()
    # scene_option.label = mujoco.mjtLabel.mjLABEL_BODY
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

    render_start_time = datetime.now()
    frames = env.render(
        traj,   # only use state.data.
        # camera="track",
        camera="track",
        scene_option=scene_option,
        width=640*2, #640
        height=480, #480
        modify_scene_fns=mod_fns,
    )
    print(f'render consumed time: {datetime.now() - render_start_time}')

    media.write_video(video_file_path,
                      frames, fps=fps)  # loop=False)
