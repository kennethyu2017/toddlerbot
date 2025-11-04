
""" follow the Joystick task for Unitree G1, project mujoco_playground."""

from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable

# from mujoco_playground._src import gait
from kbot.base_env.base_env_mjx import MjxEnv, State, Observation
from kbot.locomotion.kbot_both_leg.env_cfg import default_config
from kbot.locomotion.kbot_both_leg.joystick_reset import JoystickResetHelper

class Joystick(MjxEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if config is None:
        config = default_config(task)

    super().__init__(
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _load_keyframe(self):
      keyframe = self._mj_model.keyframe(self._config.robot.keyframes.default_pose_keyframe)
      assert keyframe.qpos.shape == (self._mj_model.nq,)
      self._init_q = jp.array(keyframe.qpos)
      self._default_pose = jp.array(
          keyframe.qpos[7:]
      )

  def _gen_soft_jnt_limit(self):
      # Note: First joint is freejoint.
      self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
      assert np.all(self._lowers < self._uppers)
      c = (self._lowers + self._uppers) / 2
      r = self._uppers - self._lowers
      self._soft_jnt_lowers = c - 0.5 * r * self._config.model.soft_joint_pos_limit_factor
      self._soft_jnt_uppers = c + 0.5 * r * self._config.model.soft_joint_pos_limit_factor

  def _find_jnt(self):

      def _get_jnt_adr(names)->npt.NDArray[np.int32]:
          adr_list=[]
          if not isinstance(names, Iterable):
              names=[names]

          dim_of_first = -1
          for _nm in names:
              jnt_view = self._mj_model.joint(_nm)
              adr = jnt_view.qposadr[0]
              dim = jnt_view.qpos0.shape[0]
              if len(adr_list) == 0:
                  dim_of_first = dim
              adr_list.extend(
                  range(adr, adr + dim)
              )
              # should have same dim.
              assert dim_of_first == dim
          # assert np.all([len(_a) == len(adr_list[0]) for _a in adr_list])
          return np.array(adr_list)

      free_jnt_adr = _get_jnt_adr(self._config.robot.joints.free_joint)
      assert free_jnt_adr == list(range(0,7))
      print(f'{free_jnt_adr=:}')

      self._hip_jnt_adr = _get_jnt_adr([*self._config.robot.joints.hip_pitch_joints,
                                        *self._config.robot.joints.hip_roll_joints])
      print(f'{self._hip_jnt_adr=:}')

      self._knee_jnt_adr = _get_jnt_adr(self._config.robot.joints.knee_pitch_joints)
      print(f'{self._knee_jnt_adr=:}')


  def _find_site(self):
      # fmt: on
      self._pelvis_imu_site_id = self._mj_model.site(self._config.robot.sites.pelvis_imu_site).id
      self._feet_site_id = np.array(
          [self._mj_model.site(name).id for name in self._config.robot.sites.feet_sites]
      )


  def _find_geom(self):
      self._floor_geom_id = self._mj_model.geom("floor").id
      self._feet_collision_geom_id = np.array(
          [self._mj_model.geom(name).id for name in self._config.robot.geoms.feet_collision_geoms]
      )
      # self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
      # self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
      # self._left_foot_geom_id = self._mj_model.geom("left_foot").id
      # self._right_foot_geom_id = self._mj_model.geom("right_foot").id
      # self._left_shin_geom_id = self._mj_model.geom("left_shin").id
      # self._right_shin_geom_id = self._mj_model.geom("right_shin").id
      # self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
      # self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id


  def _find_sensor(self):

      def _get_sensor_adr(names)->npt.NDArray[np.int32]:
          adr_list=[]
          if not isinstance(names, Iterable):
              names=[names]

          dim_of_first=-1
          for _nm in names:
              sensor_view = self._mj_model.sensor(_nm)
              adr = sensor_view.adr[0]
              dim = sensor_view.dim[0]
              if len(adr_list) == 0:
                  dim_of_first = dim
              adr_list.extend(
                  range(adr, adr + dim)
              )
              # should have same dim.
              assert dim_of_first == dim
          # assert np.all([len(_a) == len(adr_list[0]) for _a in adr_list])
          return np.array(adr_list)

      # self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
      self._feet_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_linvel_feet_ankle)
      print(f'{self._feet_linvel_sensor_adr=:}')

      self._floor_feet_found_sensor_adr = _get_sensor_adr(self._config.robot.sensors.floor_feet_found)
      print(f'{self._floor_feet_found_sensor_adr=:}')

      self._left_leg_right_leg_found_sensor_adr = _get_sensor_adr(self._config.robot.sensors.left_leg_right_leg_found)
      print(f'{self._left_leg_right_leg_found_sensor_adr=:}')

      self._feet_force_sensor_adr = _get_sensor_adr(self._config.robot.sensors.feet_force)
      print(f'{self._feet_force_sensor_adr=:}')


  def _post_init(self) -> None:
    # todo: temply comment. kenneth.
    self._load_keyframe()
    self._gen_soft_jnt_limit()
    self._find_jnt()

    # fmt: off
    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # right leg.
        1.0, 1.0, 1.0,  # waist.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm.
    ])

    self._find_site()
    self._find_geom()
    self._find_sensor()

    # self._cmd_a = jp.array(self._config.command.a)
    # self._cmd_b = jp.array(self._config.command.b)
    # TODO: how to use cmd_a, cmd_b ?
    self._cmd_a = np.array(self._config.command.a)
    self._cmd_b = np.array(self._config.command.b)

  # --- jit boundary ---
  def reset(self, rng: jax.Array) -> State:
      rng, key_data, key_cmd, key_info = jax.random.split(rng, 4)
      data:mjx.Data = JoystickResetHelper.gen_data(mj_model=self._mj_model,
                                                   mjx_model=self._mjx_model,
                                                   init_qpos=self._init_q,
                                                   init_qvel=jp.zeros(self.mjx_model.nv),
                                                   soft_lowers=self._soft_jnt_lowers,
                                                   soft_uppers=self._soft_jnt_uppers,
                                                   rng=key_data)

      cmd = self.sample_command(key_cmd)
      info = JoystickResetHelper.gen_info(rng=key_info,
                                          ctrl_dt=self.ctrl_dt,
                                          push_interval_lower=self._config.push.interval_range[0],
                                          push_interval_upper=self._config.push.interval_range[1],
                                          cmd=cmd,
                                          nu=self._mj_model.nu)

      metrics = JoystickResetHelper.gen_metrics(self._config.reward.scales.keys())

      reward, done = jp.zeros(2)

      def _gen_obs()->Observation:
          # contact = jp.array([
      	# 	data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
      	# 	for sensorid in self._feet_floor_found_sensor
      	# ])
          floor_feet_contact = data.sensordata[self._floor_feet_found_sensor_adr] > 0
          assert floor_feet_contact.shape == (2,)
          return self._get_obs(data, info, floor_feet_contact)

      # note: State must be PyTree with jp.array leaf nodes to be able to cross the jit boundary.
      return State(
          data=data,
          obs=_gen_obs(),
          reward=reward,
          done=done,
          metrics=metrics,
          info=info,
      )

  def _sample_push(self, rng:jax.Array,
                   push_step:jax.Array,
                   push_interval_steps:jax.Array)->Tuple[jax.Array,jax.Array]:
      rng, key_theta, key_mag = jax.random.split(
          rng, 3
      )
      push_theta = jax.random.uniform(key_theta, maxval=2 * jp.pi)
      push_magnitude = jax.random.uniform(
          key_mag,
          minval=self._config.push.magnitude_range[0],
          maxval=self._config.push.magnitude_range[1],
      )
      push_xy = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
      push_xy *= (
              jp.mod(push_step + 1, push_interval_steps)
              == 0
      )
      # actually we can use if enable...
      push_xy *= self._config.push.enable
      return push_xy, push_magnitude

  def _apply_push(self, state:State)-> Tuple[State, jax.Array]:
      state.info["rng"], key_push = jax.random.split(state.info["rng"])
      push_xy, push_magnitude = self._sample_push(rng=key_push,
                                                  push_step=state.info["push_step"],
                                                  push_interval_steps=state.info["push_interval_steps"])

      print(f'step() ---> sampled push_xy: {push_xy}, push_magnitude: {push_magnitude}')
      # TODO: add push to xfrc_applied : user-defined forces in joint or Cartesian coordinates
      #  (which are stored in mjData.qfrc_applied and mjData.xfrc_applied respectively).
      # mjData.xfrc_applied are Cartesian wrenches applied to the CoM of individual bodies.
      # This field is used for example, by the native viewer to apply mouse perturbations.
      qvel = state.data.qvel
      qvel = qvel.at[:2].set(push_xy * push_magnitude + qvel[:2])
      data = state.data.replace(qvel=qvel)
      state = state.replace(data=data)
      return state, push_xy

  def _handle_contact(self, state:State)->Tuple[State, jax.Array, jax.Array]:
      # contact = jp.array([
      #     state.data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
      #     for sensorid in self._feet_floor_found_sensor
      # ])

      floor_feet_contact = state.data.sensordata[self._floor_feet_found_sensor_adr] > 0

      contact_filt = floor_feet_contact | state.info["last_contact"]

      # state.info["feet_air_time"] is 0 if prev step the corresponding foot not contact with floor.
      # state.info["feet_air_time"] > 0 only if prev step the corresponding foot have contact with floor.
      first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt

      # TODO: feet_air_time record the air-time from last step to curr step of individual foot.
      # we add ctrl_dt here to record for next step, and will clear feet_air_time if curr step no foot contact with floor,
      # at following.
      state.info["feet_air_time"] += self.ctrl_dt
      return state, floor_feet_contact, first_contact

  def _update_swing_peak(self, state:State)->State:
      # xpos in world coordinate.
      p_f = state.data.site_xpos[self._feet_site_id]
      p_fz = p_f[..., -1]
      state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
      return state

  def _apply_jax_step(self, state:State, action:jax.Array)->Tuple[mjx.Data, jax.Array]:
      # TODO: clip motor targets according to soft_lower/upper ? or normalize the motor_target into [lower, upper] ?
      # NOTE: action is relative to default_pose which is from keyframe `knees_bent`.
      # i.e. normalize default_pose as 0. mean of action is output of tanh, should be in [-1, 1].
      motor_targets = self._default_pose + action * self._config.model.action_scale  # *0.5
      state = self.jax_step(
          self.mjx_model, state.data, motor_targets, self.n_substeps
      )
      return state, motor_targets


      # --- jit boundary ---
  def step(self, state: State, action: jax.Array) -> State:
    #   add push to qvel.
    state, push_xy = self._apply_push(state)

    state, motor_targets = self._apply_jax_step(state, action)

    state, floor_feet_contact, first_contact = self._handle_contact(state)
    state = self._update_swing_peak(state)

    obs = self._get_obs(state.data, state.info, floor_feet_contact)
    done = self._get_termination(state.data)

    rewards = self._get_reward(
        data=state.data,
        action=action,
        info=state.info,
        metrics=state.metrics,
        done=done,
        first_contact=first_contact,
        floor_feet_contact=floor_feet_contact
    )
    rewards = {
        k: v * self._config.reward.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.ctrl_dt

    state.info["push_xy"] = push_xy
    state.info["step"] += 1
    state.info["push_step"] += 1
    state.info["motor_targets"] = motor_targets
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # state.info["phase"] = jp.where(
    #     jp.linalg.norm(state.info["command"]) > 0.01,
    #     state.info["phase"],
    #     jp.ones(2) * jp.pi,
    # )
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )

    # TODO:   merge with state.info["feet_air_time"] += self.ctrl_dt....
    # e.g. state.info["feet_air_time"] =+  self.ctrl_dt * ~contact..
    #  no, must set state.info["feet_air_time"] to zero as sentinal for next step...
    state.info["feet_air_time"] *= ~contact

    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
    contact_termination = data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
    ] > 0
    return (
        fall_termination
        | contact_termination
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> Observation:
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,
    ])

    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self, *,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      floor_feet_contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data, "pelvis")
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data, "pelvis")
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data, "pelvis"),
            self.get_global_linvel(data, "torso"),
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            self.get_global_angvel(data, "torso")
        ),
        "orientation": self._cost_orientation(self.get_gravity(data, "torso")),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    l_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "left_foot_force"
    )
    r_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "right_foot_force"
    )
    # only penalty on contac_force > 500, i.e. jump and fall onto floor.
    cost = jp.clip(
        jp.abs(l_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    cost += jp.clip(
        jp.abs(r_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    return cost

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    c = (
        data.sensordata[
            self._mj_model.sensor_adr[self._left_hand_left_thigh_found_sensor]
        ]
        > 0
    )
    c |= (
        data.sensordata[
            self._mj_model.sensor_adr[self._right_hand_right_thigh_found_sensor]
        ]
        > 0
    )
    return jp.any(c)

  # Tracking rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    # Allow roll deviation when lateral velocity is high.
    weight = jp.where(
        cmd[1] > 0.1,
        jp.array([0.0, 1.0, 0.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    cost = jp.sum(jp.abs(error) * weight)
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    error = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
    return jp.sum(jp.abs(error))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_jnt_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_jnt_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      global_linvel_torso: jax.Array,
      global_linvel_pelvis: jax.Array,
  ) -> jax.Array:
    torso_cost = jp.square(global_linvel_torso[2])
    pelvis_cost = jp.square(global_linvel_pelvis[2])
    return torso_cost + pelvis_cost

  def _cost_ang_vel_xy(self, global_angvel_torso: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel_torso[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  # Other rewards.

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data, "pelvis")[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    del commands  # Unused.
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      command: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    body_linvel = self.get_global_linvel(data, "pelvis")[:2]
    body_angvel = self.get_global_angvel(data, "pelvis")[2]
    linvel_mask = jp.logical_or(
        jp.linalg.norm(body_linvel) > 0.1,
        jp.abs(body_angvel) > 0.1,
    )
    mask = jp.logical_or(linvel_mask, jp.linalg.norm(command) > 0.01)
    reward *= mask
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.command.lin_vel_x[0], maxval=self._config.command.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

if __name__ == "__main__":
    from kbot.locomotion.kbot_both_leg.env_cfg import default_config
    test_task = 'flat_terrain'
    test_env = Joystick(task=test_task, config=default_config(test_task))

