
""" follow the Joystick task for Unitree G1, project mujoco_playground."""

from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from jax._src.lib import pytree
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable

from mujoco_playground._src import gait
from kbot.base_env.base_env_mjx import MjxEnv, State, Observation
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

  # exclude the freejoint
  def _gen_soft_jnt_limit(self):
      # Note: jnt_range[0] is freejoint.
      self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
      assert np.all(self._lowers < self._uppers)
      c = (self._lowers + self._uppers) / 2
      r = self._uppers - self._lowers
      self._soft_jnt_lowers = c - 0.5 * r * self._config.model.soft_joint_pos_limit_factor
      self._soft_jnt_uppers = c + 0.5 * r * self._config.model.soft_joint_pos_limit_factor

  def _find_jnt(self):

      # function not in jax.jit , can use normally if else...
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

      # only need roll and yaw for cost_deviation till now.
      self._hip_r_y_jnt_adr = _get_jnt_adr([*self._config.robot.joints.hip_roll_joints,
                                            *self._config.robot.joints.hip_yaw_joints])
      print(f'{self._hip_r_y_jnt_adr=:}')

      self._knee_p_jnt_adr = _get_jnt_adr(self._config.robot.joints.knee_pitch_joints)
      print(f'{self._knee_p_jnt_adr=:}')


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

      self._pelvis_upvector_sensor_adr = _get_sensor_adr(self._config.robot.sensors.upvector_pelvis)
      print(f'{self._pelvis_upvector_sensor_adr=:}')

      # IMU
      self._pelvis_accelerometer_sensor_adr = _get_sensor_adr(self._config.robot.sensors.accelerometer_pelvis)
      print(f'{self._pelvis_accelerometer_sensor_adr=:}')

      self._pelvis_local_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.local_linvel_pelvis)
      print(f'{self._pelvis_local_linvel_sensor_adr=:}')

      self._pelvis_gyro_sensor_adr = _get_sensor_adr(self._config.robot.sensors.gyro_pelvis)
      print(f'{self._pelvis_gyro_sensor_adr=:}')

      self._pelvis_global_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_linvel_pelvis)
      print(f'{self._pelvis_global_linvel_sensor_adr=:}')

      self._pelvis_global_angvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_angvel_pelvis)
      print(f'{self._pelvis_global_angvel_sensor_adr=:}')

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
      rng, key_data, key_cmd, key_info, key_obs = jax.random.split(rng, 5)
      data:mjx.Data = JoystickResetHelper.gen_data(mj_model=self._mj_model,
                                                   mjx_model=self._mjx_model,
                                                   init_qpos=self._init_q,
                                                   init_qvel=jp.zeros(self.mjx_model.nv),
                                                   soft_lowers=self._soft_jnt_lowers,
                                                   soft_uppers=self._soft_jnt_uppers,
                                                   rng=key_data)

      # cmd = self.sample_command(key_cmd)
      # info = JoystickResetHelper.gen_info(rng=key_info,
      #                                     ctrl_dt=self.ctrl_dt,
      #                                     push_interval_lower=self._config.push.interval_range[0],
      #                                     push_interval_upper=self._config.push.interval_range[1],
      #                                     cmd=cmd,
      #                                     nu=self._mj_model.nu)

      cmd = self.sample_command(key_cmd)
      info = JoystickResetHelper.gen_info(rng=key_info,
                                          ctrl_dt=self.ctrl_dt,
                                          push_interval_lower=self._config.push.interval_range[0],
                                          push_interval_upper=self._config.push.interval_range[1],
                                          cmd=cmd,
                                          nu=self._mj_model.nu,
                                          # record the data/obs after reset, then used when step() `done` through soft-reset mechanism.
                                          # first_data=data,
                                          # first_obs=obs )
                                          )

      def _gen_obs() -> Observation:
          # contact = jp.array([
          # 	data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
          # 	for sensorid in self._feet_floor_found_sensor
          # ])
          floor_feet_contact = data.sensordata[self._floor_feet_found_sensor_adr] > 0
          assert floor_feet_contact.shape == (2,)
          return self._get_obs(
              rng=key_obs,
              data=data,
              floor_feet_contact=floor_feet_contact,
              feet_air_time=info['feet_air_time'],
              phase=info['phase'],
              curr_act=info['last_act'],
              last_act=info['last_last_act'],
              cmd=cmd)

      obs = _gen_obs()

      metrics = JoystickResetHelper.gen_metrics(self._config.reward.scales.keys())
      reward, done = jp.zeros(2)

      # note: State must be PyTree with jp.array leaf nodes to be able to cross the jit boundary.
      return State(
          data=data,
          obs=obs,
          reward=reward,
          done=done,
          metrics=metrics,
          info=info,
          # kenneth: record for soft reset.
          reset_data=data,
          reset_obs=obs,
          reset_info=info
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

  def _apply_push(self, state:State, rng:jax.Array)-> State:
      # state.info["rng"], key_push = jax.random.split(state.info["rng"])
      push_xy, push_magnitude = self._sample_push(rng=rng,
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

      # record.
      state.info["push_xy"] = push_xy
      state.info["push_step"] += 1

      return state

  def _handle_contact(self, state:State)->Tuple[State, jax.Array, jax.Array, jax.Array]:
      # contact = jp.array([
      #     state.data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
      #     for sensorid in self._feet_floor_found_sensor
      # ])

      floor_feet_contact = state.data.sensordata[self._floor_feet_found_sensor_adr] > 0

      # TODO: feet_air_time accumulate the air-time from prev un-contact step un-till curr step of individual foot.
      # add ctrl_dt if not consecutive contact, and will clear feet_air_time if curr step no foot contact with floor, before exit from step().
      # case 4:
      consecutive_contact = (state.info["last_contact"] * floor_feet_contact)

      # add ctrl_dt when not case 4.
      state.info["feet_air_time"] += (self.ctrl_dt * ~consecutive_contact)

      # contact_filt = floor_feet_contact | state.info["last_contact"]

      # state.info["feet_air_time"] is 0 if prev step the corresponding foot not contact with floor.
      # state.info["feet_air_time"] > 0 only if prev step the corresponding foot have contact with floor.
      # first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt

      # kenneth:last step no contact and curr step has contact, then it is `first contact` .
      # even this is the first step after reset, state.info["last_contact"] is True, so first_contact is False here.
      # case 2.
      first_contact = ~state.info["last_contact"] * floor_feet_contact

      # # TODO: feet_air_time accumulate the air-time from prev un-contact step un-till curr step of individual foot.
      # # will clear feet_air_time if curr step no foot contact with floor, at following.
      # state.info["feet_air_time"] += self.ctrl_dt

      accumulate_air_time = state.info["feet_air_time"]

      # clear running accumulated info['feet_air_time'] if contact with floor.
      state.info["feet_air_time"] *= ~floor_feet_contact
      state.info["last_contact"] = floor_feet_contact

      return state, floor_feet_contact, first_contact, accumulate_air_time

  # kenneth: collect swing peak only when foot in the air.
  def _update_swing_peak(self, state:State, floor_feet_contact: jax.Array)->Tuple[State, jax.Array]:
      # xpos in world coordinate.
      p_f = state.data.site_xpos[self._feet_site_id]
      p_fz = p_f[..., -1]
      state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
      swing_peak_in_air = state.info["swing_peak"]

      # clear running statistics.
      state.info["swing_peak"] *= ~floor_feet_contact

      return state, swing_peak_in_air

  def _apply_jax_step(self, state:State, action:jax.Array)->Tuple[State, jax.Array, jax.Array]:
      # TODO: clip motor targets according to soft_lower/upper ? or normalize the motor_target into [lower, upper] ?
      # NOTE: action is relative to default_pose which is from keyframe `knees_bent`.
      # i.e. normalize default_pose as 0. mean of action is output of tanh, should be in [-1, 1].
      motor_targets = self._default_pose + action * self._config.model.action_scale  # *0.5
      new_data = self.jax_step(
          self.mjx_model, state.data, motor_targets, self.n_substeps
      )
      state=state.replace(data=new_data)

      last_last_act = state.info["last_last_act"]
      last_act = state.info["last_act"]

      # update
      state.info["motor_targets"] = motor_targets
      state.info["last_last_act"] = state.info["last_act"]
      state.info["last_act"] = action

      return state, last_last_act, last_act

  @staticmethod
  def _update_phase(state:State)->Tuple[State, jax.Array, jax.Array]:
      phase_tp1 = state.info["phase"] + state.info["phase_dt"]

      # kenneth: obs will use updated phase, reward use last phase.
      last_phase = state.info["phase"]

      # kenneth: map phase from [ 0, 2pi] -> [-pi, pi].
      state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

      # NOTE(kevin): Enable this to make the policy stand still at 0 command.
      # state.info["phase"] = jp.where(
      #     jp.linalg.norm(state.info["command"]) > 0.01,
      #     state.info["phase"],
      #     jp.ones(2) * jp.pi,
      # )
      new_phase = state.info["phase"]

      return state, last_phase, new_phase


  def _update_cmd(self, state:State,
                  rng:jax.Array
                  )->Tuple[State, jax.Array,jax.Array]:
      last_cmd = state.info["command"]

      # state.info["command"] = jp.where(
      #     state.info["step"] > 500,
      #     self.sample_command(rng),
      #     state.info["command"],
      # )
      # kenneth: we re-sample command after done, cause in AutoResetWrapper, will not re-sample
      # command after `done`.
      # TODO: after done, even we re-sample command here,  AutoResetWrapper will not use
      # the obs we returned, instead, it will use the reset_obs which include the command
      # from env first reset.
      # BUG: so, the obs and info["command"] is not aligned....
      #

      curr_step = state.info["resample_cmd_steps"]
      curr_step += 1

      state.info["command"] = jp.where(
              # state.done | curr_step > 500,
              curr_step > self._config.command.resample_length, #500,
              self.sample_command(rng),
              state.info["command"],
          )
      new_cmd = state.info["command"]

      # reset if beyond 500.
      # TODO: NOTE, we don't clear resample_cmd_steps after done, cause in the beginning
      # of training, a lot of `done` happen, we don't need to do too much re-sample.
      # always re-sample according to simu-steps.
      state.info["resample_cmd_steps"] = jp.where(
          # state.done | (curr_step > 500)
          curr_step > self._config.command.resample_length, # 500,
          0,
          curr_step,
      )

      return state, last_cmd, new_cmd

  # @staticmethod
  # def _update_step_count(state:State)->Tuple[State, jax.Array]:
  #     state.info["resample_cmd_steps"] += 1
  #     curr_step = state.info["resample_cmd_steps"]
  #
  #     # kenneth: NOTE: in EpisodeWrapper/AutoResetWrapper, info use `steps`, no same as here "step".
  #     state.info["resample_cmd_steps"] = jp.where(
  #         state.done | (state.info["resample_cmd_steps"] > 500),
  #         0,
  #         state.info["resample_cmd_steps"],
  #     )
  #     return state, curr_step


  # update new_cmd into reset_obs. for AutoResetWrapper.
  @staticmethod
  def _update_reset_obs(state:State, new_cmd:jax.Array, cmd_idx:Dict[str,slice])->State:
      assert new_cmd.shape == (3,)
      # first obs after reset.
      # first_obs:Observation = state.info["first_obs"]
      reset_obs:Observation = state.reset_obs

      def _replace_cmd(path:Tuple[pytree.DictKey], x:jax.Array) -> jax.Array:
          # key is 'state', 'privileged_state'.
          key = path[0].key
          idx:slice = cmd_idx[key]
          return x.at[idx].set(new_cmd)

      # update cmd to new_cmd in case re-sample.
      reset_obs = jax.tree.map_with_path(_replace_cmd, reset_obs)

      # update cmd to curr command which maybe re-sampled every 500-step.
      # NOTE: the cmd index must be 0:3
      # reset_obs = first_obs['state'].at[cmd_idx_slice].set(new_cmd)

      state = state.replace(reset_obs=reset_obs)

      return state

  # for AutoResetWrapper
  @staticmethod
  def _update_reset_info(state:State, new_cmd: jax.Array)->State:
      # in-place update rng, cmd. no need to call state.replace(reset_info=...)
      state.reset_info.update({
          # always deliver new rng after reset.
          "rng": state.info["rng"],
          "command": new_cmd,

          # TODO: re-sample phase_dt?
          # Phase related.
          # "phase_dt": phase_dt,
          # "phase": phase,  # [0, pi]

          # TODO: re-sample Push ?
          # "push_xy": jp.array([0.0, 0.0]),
          # "push_step": 0,
          # "push_interval_steps": push_interval_steps,

          # record the data/obs after reset, then used when step() `done` through soft-reset mechanism.
          # 'first_data': first_data,
          # NOTE: in first_obs, the  "command" maybe different as step() used when `done`, cause the
          # command will be re-sampled every 500-steps in step().
          # 'first_obs': first_obs,
      })

      # TODO: we can not replace state.info which also include the outter-wrapper specific key/values.
      # state=state.replace(reset_info=reset_info)

      return state


  @staticmethod
  def _update_metrics(state:State, rewards: Dict[str, jax.Array], swing_peak_in_air:jax.Array)->State:
      for k, v in rewards.items():
          state.metrics[f"reward/{k}"] = v

      # kenneth
      # state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
      state.metrics["swing_peak"] = jp.mean(swing_peak_in_air)
      return state


  # --- jit boundary ---
  # TODO:NOTE: we can not replace state.info which also include the outter-wrapper specific key/values.
  def step(self, state: State, action: jax.Array) -> State:
     # after we update rng, we will update new rng into reset_info also.
     state.info["rng"], key_push, key_obs, key_cmd = jax.random.split(state.info["rng"], 4)

    # kenneth: flax.struct.dataclass is a 'frozen' dataclass, so must use .replace() to modify member.
    # note: .replace() do shallow copy on other non_replaced data members, so it is efficient
    # to do .replace() multiple times in each child functions here, especially used in jit.

    # add push to qvel.
     state = self._apply_push(state, key_push)
     state, last_last_act, last_act= self._apply_jax_step(state, action)

     state, floor_feet_contact, first_contact, feet_air_time = self._handle_contact(state)
     state, swing_peak_in_air = self._update_swing_peak(state, floor_feet_contact)
     state, last_phase, new_phase = self._update_phase(state)

     # kenneth: the BraxAutoResetWrapper will make use of done to reset env.
     state = self._update_termination(state)
     # done = done.astype(reward.dtype)
     # state = state.replace(done=done)

     # state, curr_step = self._update_step_count(state)
     # will re-sample cmd if `done` or accumulate 500 steps.
     state, last_cmd, new_cmd = self._update_cmd(state, key_cmd)

     # kenneth: obs is for next step, so we should use some updated data.
     # will update state.obs, state.reset_obs.
     state = self._update_obs(
        rng=key_obs,
        state=state,
        floor_feet_contact=floor_feet_contact,
        feet_air_time=feet_air_time,
        # kenneth: use updated phase and cmd in obs for next step.
        phase=new_phase,

        # kenneth: we should use record curr_act and last_act in obs for next step.
        # last_act= info["last_act"] )
        curr_act=action,
        last_act=last_act,

        # TODO: after done, even we re-sample command here,  AutoResetWrapper will not use
        # the obs we returned, instead, it will use the reset_obs which include the command
        # from env first reset.
        # BUG: so, the obs and info["command"] is not aligned....
        cmd=new_cmd,
     )

     # TODO: maybe we can use last obs (in state arg) to get some info to be used in rwd calc, instead
     # recording so much stuff in state.info[].
     # kenneth: rwd will compare some info of last step with result(in state.data) of curr step.
     rewards = self._get_rewards(
        data=state.data,
        curr_act=action,
        last_act=last_act,
        last_last_act=last_last_act,
        done=state.done,
        first_contact=first_contact,
        floor_feet_contact=floor_feet_contact,
        feet_air_time=feet_air_time,
        swing_peak_in_air=swing_peak_in_air,
        last_cmd=last_cmd,
        last_phase=last_phase,
     )
     rewards = {
        k: v * self._config.reward.scales[k] for k, v in rewards.items()
     }
     reward = sum(rewards.values()) * self.ctrl_dt
     state = state.replace(reward=reward)

     state = self._update_metrics(state, rewards, swing_peak_in_air)

     # for AutoResetWrapper.
     state = self._update_reset_info(state, new_cmd)

     # Finally, we do the soft-reset like in BraxAutoResetWrapper to replace data/obs only instead of
     # calling the env.reset().
     # output_obs = jp.where(state.done, soft_reset_obs, curr_obs)
     # state = state.replace(obs=output_obs)
     # output_data = jp.where(state.done, state.info["first_data"], state.data)
     # state = state.replace(data=output_data)

    # state.info["step"] += 1
    # state.info["push_xy"] = push_xy
    # state.info["push_step"] += 1
    # state.info["motor_targets"] = motor_targets

    # phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    #
    # # kenneth: map phase from [ 0, 2pi] -> [-pi, pi].
    # state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # # state.info["phase"] = jp.where(
    # #     jp.linalg.norm(state.info["command"]) > 0.01,
    # #     state.info["phase"],
    # #     jp.ones(2) * jp.pi,
    # # )

    # state.info["last_last_act"] = state.info["last_act"]
    # state.info["last_act"] = action

    # state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    # state.info["command"] = jp.where(
    #     state.info["step"] > 500,
    #     self.sample_command(cmd_rng),
    #     state.info["command"],
    # )
    # state.info["step"] = jp.where(
    #     done | (state.info["step"] > 500),
    #     0,
    #     state.info["step"],
    # )

    # TODO:   merge with state.info["feet_air_time"] += self.ctrl_dt....
    # e.g. state.info["feet_air_time"] =+  self.ctrl_dt * ~contact..
    #  no, must set state.info["feet_air_time"] to zero as sentinal for next step...
    # state.info["feet_air_time"] *= ~contact
    # state.info["last_contact"] = contact
    # state.info["swing_peak"] *= ~contact

    # for k, v in rewards.items():
    #   state.metrics[f"reward/{k}"] = v
    #
    # # kenneth
    # # state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    # state.metrics["swing_peak"] = jp.mean(swing_peak_in_air)

    # done = done.astype(reward.dtype)
    # state = state.replace(data=data, obs=obs, reward=reward, done=done)
    # state = state.replace(obs=obs, reward=reward, done=done)
     return state


  def _update_termination(self, state: State) -> State:
    # z axis should be along world coordinate.
    fall_termination = state.data.sensordata[self._pelvis_upvector_sensor_adr][-1] < 0.0

    contact_termination = jp.any(state.data.sensordata[ self._left_leg_right_leg_found_sensor_adr] > 0)

    done = (
            fall_termination
            | contact_termination
            | jp.isnan(state.data.qpos).any()
            | jp.isnan(state.data.qvel).any()
    )
    # done=done.astype(reward.dtype)
    state = state.replace(done=done)
    return state



  # def _get_termination(self, data: mjx.Data) -> jax.Array:
  #   fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
  #
  #   contact_termination = data.sensordata[
  #       self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
  #   ] > 0
  #   contact_termination |= data.sensordata[
  #       self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
  #   ] > 0
  #   contact_termination |= data.sensordata[
  #       self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
  #   ] > 0
  #   return (
  #       fall_termination
  #       | contact_termination
  #       | jp.isnan(data.qpos).any()
  #       | jp.isnan(data.qvel).any()
  #   )

  def _get_obs(self, *,
               rng: jax.Array,
               data: mjx.Data,
               floor_feet_contact: jax.Array,
               feet_air_time: jax.Array,
               phase: jax.Array,
               curr_act: jax.Array,
               last_act: jax.Array,
               cmd: jax.Array,
               )->Observation:
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      rng, key_gyro, key_gravity, key_qpos, key_qvel, key_linvel = jax.random.split(rng, 6)

      # gyro = self.get_gyro(data, "pelvis")
      gyro = data.sensordata[self._pelvis_gyro_sensor_adr]
      noisy_gyro = (
              gyro
              + (2 * jax.random.uniform(key_gyro, shape=gyro.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.gyro
      )

      gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_gravity = (
              gravity
              + (2 * jax.random.uniform(key_gravity, shape=gravity.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.gravity
      )

      joint_angles = data.qpos[7:]
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_angles = (
              joint_angles
              + (2 * jax.random.uniform(key_qpos, shape=joint_angles.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.joint_pos
      )

      joint_vel = data.qvel[6:]
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_joint_vel = (
              joint_vel
              + (2 * jax.random.uniform(key_qvel, shape=joint_vel.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.joint_vel
      )

      # cos = jp.cos(info["phase"])
      # sin = jp.sin(info["phase"])
      cos = jp.cos(phase)
      sin = jp.sin(phase)
      phase = jp.concatenate([cos, sin])

      # linvel = self.get_local_linvel(data, "pelvis")
      linvel = data.sensordata[self._pelvis_local_linvel_sensor_adr]
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_linvel = (
              linvel
              + (2 * jax.random.uniform(key_linvel, shape=linvel.shape) - 1)
              * self._config.noise.level
              * self._config.noise.scales.linvel
      )

      # TODO: check all obs values can be got on real robot through sensors.
      policy_state = jp.hstack([
          # # info["command"], # 3
          cmd,  # 3
          noisy_linvel,  # 3
          noisy_gyro,  # 3
          noisy_gravity,  # 3
          # # # info["command"], # 3
          # cmd,               # 3
          noisy_joint_angles - self._default_pose,  # 29
          noisy_joint_vel,  # 29
          # kenneth: we should record curr_act and last_act in obs for next step.
          # info["last_act"],  # 29
          curr_act,  # 29
          last_act,  # 29
          phase,
      ])

      # accelerometer = self.get_accelerometer(data, "pelvis")
      # global_angvel = self.get_global_angvel(data, "pelvis")

      accelerometer = data.sensordata[self._pelvis_accelerometer_sensor_adr]
      global_angvel = data.sensordata[self._pelvis_global_angvel_sensor_adr]

      feet_vel = data.sensordata[self._feet_linvel_sensor_adr].ravel()
      root_height = data.qpos[2]

      privileged_state = jp.hstack([
          policy_state,
          gyro,  # 3
          accelerometer,  # 3
          gravity,  # 3
          linvel,  # 3
          global_angvel,  # 3
          joint_angles - self._default_pose,
          joint_vel,
          root_height,  # 1
          data.actuator_force,  # 29
          floor_feet_contact,  # 2
          feet_vel,  # 4*3
          # info["feet_air_time"],  # 2
          feet_air_time,  # 2
      ])

      curr_obs = {
          # input for policy network.
          "state": policy_state,
          # input for value network.
          "privileged_state": privileged_state,
      }

      return curr_obs



  # will update state.obs, state.reset_obs.
  def _update_obs(
      self,*,
          rng: jax.Array,
          state: State,
          # info: dict[str, Any],
          floor_feet_contact: jax.Array,
          feet_air_time: jax.Array,
          phase: jax.Array,
          curr_act: jax.Array,
          last_act: jax.Array,
          cmd: jax.Array,
  ) -> State:

    curr_obs = self._get_obs(
        rng=rng,
        data=state.data,
        floor_feet_contact=floor_feet_contact,
        feet_air_time=feet_air_time,
        phase=phase,
        curr_act=curr_act,
        last_act=last_act,
        cmd=cmd,
    )

    state = state.replace(obs=curr_obs)

    # we do the soft-reset like in BraxAutoResetWrapper to replace data/obs only instead of
    # calling the env.reset(), with update new_cmd into first_obs.
    # soft_reset_obs = self._gen_soft_reset_obs(
    state = self._update_reset_obs(
        state=state,
        new_cmd=cmd,
        # idx of `cmd` in policy_state and privileged_state:
        cmd_idx={'state': slice(0, 3), 'privileged_state':slice(0, 3)}
      )

    return state

  def _get_rewards(
      self, *,
      data: mjx.Data,
      curr_act: jax.Array,
      last_act: jax.Array,
      last_last_act: jax.Array,
      done: jax.Array,
      first_contact: jax.Array,
      floor_feet_contact: jax.Array,
      feet_air_time: jax.Array,
      swing_peak_in_air: jax.Array,
      last_cmd: jax.Array,
      last_phase: jax.Array
  ) -> Dict[str, jax.Array]:
    # del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            # info["command"], self.get_local_linvel(data, "pelvis")
            last_cmd,
            # self.get_local_linvel(data, "pelvis")
            data.sensordata[self._pelvis_local_linvel_sensor_adr]
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            # info["command"], self.get_gyro(data, "pelvis")
            last_cmd,
            # self.get_gyro(data, "pelvis")
            data.sensordata[self._pelvis_gyro_sensor_adr]
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            # self.get_global_linvel(data, "pelvis"),
            # self.get_global_linvel(data, "torso"),
            data.sensordata[self._pelvis_global_linvel_sensor_adr],
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            # self.get_global_angvel(data, "torso")
            data.sensordata[self._pelvis_global_angvel_sensor_adr],
        ),

        "orientation": self._cost_orientation(
            # self.get_gravity(data, "torso")
            data.sensordata[self._pelvis_upvector_sensor_adr],
        ),

        # "base_height": self._cost_base_height(data.qpos[2]),
        "base_height": self._cost_base_height(data),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            # action, info["last_act"], info["last_last_act"]
            curr_act, last_act, last_last_act
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, floor_feet_contact),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            # info["swing_peak"], first_contact, info
            swing_peak_in_air, first_contact
        ),
        "feet_air_time": self._reward_feet_air_time(
            # info["feet_air_time"], first_contact, info["command"]
            # kenneth: use the air time output from _handle_contact, cause we will clear
            # info["feet_air_time"] in handle_contact.
            # feet_air_time, first_contact, info["command"]
            feet_air_time, first_contact, last_cmd
        ),
        "feet_phase": self._reward_feet_phase(
            # data,
            # info["phase"],
            # self._config.reward.max_foot_height,
            # info["command"],
            data, last_phase, self._config.reward.max_foot_height, last_cmd
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        # "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        # "stand_still": self._cost_stand_still(last_cmd, data.qpos[7:]),
        "stand_still": self._cost_stand_still(data, last_cmd),
        "hand_collision": self._cost_hand_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            # data.qpos[7:], info["command"]
            data, last_cmd
        ),
        # "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data),
        # "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data),
        # "pose": self._cost_pose(data.qpos[7:]),
        "pose": self._cost_pose(data),
    }

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    # l_contact_force = mjx_env.get_sensor_data(
    #     self.mj_model, data, "left_foot_force"
    # )

    # both feet, ndim=3 contact frc.
    feet_contact_frc = data.sensordata[self._feet_force_sensor_adr]

    # r_contact_force = mjx_env.get_sensor_data(
    #     self.mj_model, data, "right_foot_force"
    # )

    l_z_frc, r_z_frc = feet_contact_frc[jp.array([2, 5])]

    # only penalty on contac_force > 500, i.e. jump and fall onto floor.
    cost = jp.clip(
        # jp.abs(l_contact_force[2])
        jp.abs(l_z_frc)
        - self._config.reward.max_contact_force,
        min=0.0,
    )
    cost += jp.clip(
        # jp.abs(r_contact_force[2])
        jp.abs(r_z_frc)
        - self._config.reward.max_contact_force,
        min=0.0,
    )
    return cost

  def _cost_hand_collision(self, data: mjx.Data) -> jax.Array:
    # c = (
    #     data.sensordata[
    #         self._mj_model.sensor_adr[self._left_hand_left_thigh_found_sensor]
    #     ]
    #     > 0
    # )
    # c |= (
    #     data.sensordata[
    #         self._mj_model.sensor_adr[self._right_hand_right_thigh_found_sensor]
    #     ]
    #     > 0
    # )
    # return jp.any(c)
    return jp.array(.0)

  # Tracking rewards.

  # penalty on hip roll/yaw.
  def _cost_joint_deviation_hip(
      self, data:mjx.Data, cmd: jax.Array
  ) -> jax.Array:
    # hip roll (l,r), hip yaw (l,r) :
    # error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    error = data.qpos[self._hip_r_y_jnt_adr] - self._init_q[self._hip_r_y_jnt_adr]

    # Allow roll deviation when lateral velocity is high.
    weight = jp.where(
        cmd[1] > 0.1,
        # left - r y, right - r y,
        # jp.array([0.0, 1.0, 0.0, 1.0]),

        # hip roll (l,r), hip yaw (l,r) :
        jp.array([0.0, 0.0, 1.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    cost = jp.sum(jp.abs(error) * weight)
    return cost

  def _cost_joint_deviation_knee(self, data:mjx.Data) -> jax.Array:
    # error = qpos[self._knee_p_jnt_adr] - self._default_pose[self._knee_p_jnt_adr]
    error = data.qpos[self._knee_p_jnt_adr] - self._init_q[self._knee_p_jnt_adr]
    return jp.sum(jp.abs(error))

  def _cost_pose(self, data:mjx.Data) -> jax.Array:
    # return jp.sum(jp.square(qpos - self._default_pose))
    return jp.sum(jp.square(data.qpos[7:] - self._default_pose))

  # exclude the freejnt
  def _cost_joint_pos_limits(self, data:mjx.Data) -> jax.Array:
    out_of_limits = -jp.clip(data.qpos[7:] - self._soft_jnt_lowers, None, 0.0)
    out_of_limits += jp.clip(data.qpos[7:] - self._soft_jnt_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      # global_linvel_torso: jax.Array,
      global_linvel_pelvis: jax.Array,
  ) -> jax.Array:
    # torso_cost = jp.square(global_linvel_torso[2])
    pelvis_cost = jp.square(global_linvel_pelvis[2])
    # return torso_cost + pelvis_cost
    return pelvis_cost


  def _cost_ang_vel_xy(self,
                       # global_angvel_torso: jax.Array
                       global_angvel_pelvis: jax.Array
                       ) -> jax.Array:
    # return jp.sum(jp.square(global_angvel_torso[:2]))
    return jp.sum(jp.square(global_angvel_pelvis[:2]))

  def _cost_orientation(self,
                        # torso_zaxis: jax.Array
                        pelvis_zaxis: jax.Array
                        ) -> jax.Array:
    # return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))
    # TODO:  jp.array([0.073, 0.0, 1.0]) is read from sensordata after set to default_pose?
    return jp.sum(jp.square(pelvis_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_base_height(self, data:mjx.Data) -> jax.Array:
    return jp.square(
        data.qpos[2] - self._config.reward.base_height_target
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
      self, data:mjx.Data, commands: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(data.qpos[7:] - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, floor_feet_contact: jax.Array
  ) -> jax.Array:
    # body_vel = self.get_global_linvel(data, "pelvis")[:2]
    body_vel = data.sensordata[self._pelvis_global_linvel_sensor_adr][:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * floor_feet_contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data
  ) -> jax.Array:
    feet_vel = data.sensordata[self._feet_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
  ) -> jax.Array:
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

    # body_linvel = self.get_global_linvel(data, "pelvis")[:2]
    # body_angvel = self.get_global_angvel(data, "pelvis")[2]

    body_linvel = data.sensordata[self._pelvis_global_linvel_sensor_adr][:2]
    # around global z-axis
    body_angvel = data.sensordata[self._pelvis_global_angvel_sensor_adr][2]

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

