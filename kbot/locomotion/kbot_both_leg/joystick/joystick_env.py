
""" follow the Joystick task for Unitree G1, project mujoco_playground."""

from typing import Any, Dict, Optional, Union, Tuple, List

import jax
import jax.numpy as jp
from jax._src.lib import pytree
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
import numpy.typing as npt

from kbot.base_env.base_env_mjx import MjxEnv, State, Observation
from kbot.locomotion.kbot_both_leg.env_cfg import default_config
from kbot.locomotion.kbot_both_leg.joystick.joystick_reset import JoystickResetHelper

# from kbot.locomotion.kbot_both_leg.joystick.joystick_rwd import JoystickReward
# !!!!!!  temply for debug !!!!!!
from kbot.locomotion.kbot_both_leg.joystick.try_g1_joystick_rwd import JoystickReward

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

    self._rwd_handler = JoystickReward(
        joint_adr=self._joint_adr,
        sensor_adr=self._sensor_adr,
        site_id=self._site_id,
        rwd_cfg=self._config.reward,
    )


  def _load_keyframe(self):
      keyframe = self._mj_model.keyframe(self._config.robot.keyframes.default_pose_keyframe)
      print(f'load keyframe: {keyframe}')
      assert keyframe.qpos.shape == (self._mj_model.nq,)
      # NOTE: init_q including the freejoint.
      self._init_q = jp.array(keyframe.qpos)
      print(f'set init_q to: {self._init_q}')

      # self._default_pose = jp.array(
      #     keyframe.qpos[7:]
      # )
      # print(f'set default pose to: {self._default_pose}')

  # exclude the freejoint
  def _gen_soft_jnt_range(self)->List[jax.Array]:
      # Note: jnt_range[0] is freejoint.
      _lowers, _uppers = self._mj_model.jnt_range[1:].T
      assert np.all(_lowers < _uppers)
      c = (_lowers + _uppers) / 2
      r = _uppers -  _lowers
      soft_lowers = c - 0.5 * r * self._config.model.soft_joint_pos_limit_factor
      soft_uppers = c + 0.5 * r * self._config.model.soft_joint_pos_limit_factor
      return [soft_lowers, soft_uppers]

  def _find_jnt(self)->config_dict.FrozenConfigDict:

      # function not in jax.jit , can use normally if else...
      def _get_jnt_adr(names)->npt.NDArray[np.int32]:
          adr_list=[]

          if not isinstance(names, list or tuple):
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
      assert np.all(free_jnt_adr == list(range(0,7)) )
      # print(f'{free_jnt_adr=:}')

      # print(f'{self._hip_r_y_jnt_adr=:}')
      # print(f'{self._knee_p_jnt_adr=:}')

      return config_dict.FrozenConfigDict(
          config_dict.create(
              # only need roll and yaw for cost_deviation till now.
              hip_r_y_jnt_adr=_get_jnt_adr([*self._config.robot.joints.hip_roll_joints,
                                        *self._config.robot.joints.hip_yaw_joints]),
              knee_p_jnt_adr=_get_jnt_adr(self._config.robot.joints.knee_pitch_joints),
          )
      )


  def _find_site(self)->config_dict.FrozenConfigDict:
      return config_dict.FrozenConfigDict(
          config_dict.create(
              # fmt: on
              pelvis_imu_site_id = self._mj_model.site(self._config.robot.sites.pelvis_imu_site).id,
              feet_site_id = np.array(
                  [self._mj_model.site(name).id for name in self._config.robot.sites.feet_sites]
              )
          )
      )


  def _find_geom(self)->config_dict.FrozenConfigDict:
      return config_dict.FrozenConfigDict(
          config_dict.create(
              floor_geom_id = self._mj_model.geom("floor").id,
              feet_collision_geom_id = np.array(
                  [self._mj_model.geom(name).id for name in self._config.robot.geoms.feet_collision_geoms]
              )
          )
      )
      # self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
      # self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
      # self._left_foot_geom_id = self._mj_model.geom("left_foot").id
      # self._right_foot_geom_id = self._mj_model.geom("right_foot").id
      # self._left_shin_geom_id = self._mj_model.geom("left_shin").id
      # self._right_shin_geom_id = self._mj_model.geom("right_shin").id
      # self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
      # self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id


  def _find_sensor(self)->config_dict.FrozenConfigDict:

      def _get_sensor_adr(names)->npt.NDArray[np.int32]:
          adr_list=[]
          if not isinstance(names, list or tuple):
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

      return config_dict.FrozenConfigDict(
          config_dict.create(
              pelvis_upvector_sensor_adr = _get_sensor_adr(self._config.robot.sensors.upvector_pelvis),
              # IMU
              pelvis_accelerometer_sensor_adr = _get_sensor_adr(self._config.robot.sensors.accelerometer_pelvis),
              pelvis_local_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.local_linvel_pelvis),
              pelvis_gyro_sensor_adr = _get_sensor_adr(self._config.robot.sensors.gyro_pelvis),

              pelvis_global_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_linvel_pelvis),
              pelvis_global_angvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_angvel_pelvis),
              feet_linvel_sensor_adr = _get_sensor_adr(self._config.robot.sensors.global_linvel_feet_ankle),
              floor_feet_found_sensor_adr = _get_sensor_adr(self._config.robot.sensors.floor_feet_found),
              left_leg_right_leg_found_sensor_adr = _get_sensor_adr(self._config.robot.sensors.left_leg_right_leg_found),
              feet_force_sensor_adr = _get_sensor_adr(self._config.robot.sensors.feet_force),
          )
      )


  def _find_body(self)->config_dict.FrozenConfigDict:
      return config_dict.FrozenConfigDict(
          config_dict.create(
              virtual_floating_base_body_id = self._mj_model.body(self._config.robot.bodies.virtual_floating_base).id,
              # left, right.
              pelvis_body_id = np.array(
                  [self._mj_model.body(name).id for name in self._config.robot.bodies.pelvis]
              )))

  def _post_init(self) -> None:
    # todo: temply comment. kenneth.
    self._load_keyframe()

    print(f'find robot model --->')

    self._soft_joint_range = self._gen_soft_jnt_range()
    print(f'soft joint range: {self._soft_joint_range}')
    assert np.all(self._soft_joint_range[0] < self._soft_joint_range[1])

    # qpos adr.
    self._joint_adr = self._find_jnt()
    print(f'joint adr: {self._joint_adr.to_json_best_effort()}')

    # fmt: off.
    # used for cost pose.
    # allow hip pitch , knee pitch.
    # TODO: check again.
    self._cost_pose_weight = jp.array([
        0.01, 0.01, # shoulder.
        0.01, 1.0, 1.0, 0.01, 1.0, #1.0,  # left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, #1.0,  # right leg.
        # 1.0, 1.0, 1.0,  # waist.
        # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm.
        # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm.
    ])

    self._body_id = self._find_body()
    print(f'body id: {self._body_id.to_json_best_effort()}')

    self._site_id = self._find_site()
    print(f'site id: {self._site_id.to_json_best_effort()}')

    self._geom_id = self._find_geom()
    print(f'geom_id: {self._geom_id.to_json_best_effort()}')

    self._sensor_adr = self._find_sensor()
    print(f'sensor adr: {self._sensor_adr.to_json_best_effort()}')

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
                                                   soft_joint_range=self._soft_joint_range,
                                                   rng=key_data)

      # cmd = self.sample_command(key_cmd)
      # info = JoystickResetHelper.gen_info(rng=key_info,
      #                                     ctrl_dt=self.ctrl_dt,
      #                                     push_interval_lower=self._config.push.interval_range[0],
      #                                     push_interval_upper=self._config.push.interval_range[1],
      #                                     cmd=cmd,
      #                                     nu=self._mj_model.nu)

      cmd = self.sample_command(key_cmd)
      mjxenv_info = JoystickResetHelper.gen_info(rng=key_info,
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
          floor_feet_contact = data.sensordata[self._sensor_adr.floor_feet_found_sensor_adr] > 0
          assert floor_feet_contact.shape == (2,)
          return self._get_obs(
              rng=key_obs,
              data=data,
              floor_feet_contact=floor_feet_contact,
              feet_air_time=mjxenv_info['feet_air_time'],
              phase=mjxenv_info['phase'],
              curr_act=mjxenv_info['last_act'],
              last_act=mjxenv_info['last_last_act'],
              cmd=cmd)

      obs = _gen_obs()

      metrics = JoystickResetHelper.gen_metrics(self._config.reward.scales.keys())
      reward = jp.zeros((), dtype=float)
      done = jp.zeros((), dtype=bool)

      # note: State must be PyTree with jp.array leaf nodes to be able to cross the jit boundary.
      return State(
          data=data,
          obs=obs,
          reward=reward,
          done=done,
          metrics=metrics,

          # kenneth: used by outer wrapper: EpisodeWrapper, AutoResetWrapper, EvalWrapper, etc.
          info={},

          mjxenv_info=mjxenv_info,
          # kenneth: record for soft reset.
          reset_data=data,
          reset_obs=obs,
          reset_mjxenv_info=mjxenv_info
      )

  def _sample_push(self, rng:jax.Array,
                   # push_step:jax.Array,
                   # push_interval_steps:jax.Array
                   )->Tuple[jax.Array,jax.Array]:
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

      # push_xy *= (
      #         jp.mod(push_step + 1, push_interval_steps)
      #         == 0
      # )
      # push_xy = jp.where(
      #     push_step > push_interval_steps,
      #     push_xy,
      #     0)

      # # actually we can use if enable...
      # push_xy *= self._config.push.enable

      return push_xy, push_magnitude

  def _apply_push(self, state:State, rng:jax.Array)-> State:
      # state.mjxenv_info["rng"], key_push = jax.random.split(state.mjxenv_info["rng"])

      # let jit compilation select the correct execution path.
      if not self._config.push.enable:
        return state

      push_step = state.mjxenv_info["push_step"]
      push_step += 1
      push_xy, push_magnitude = self._sample_push(rng=rng,
                                                  # push_step=push_step,
                                                  # push_interval_steps=push_interval_steps
                                                  )

      # actually we can use if enable...
      # push_xy *= self._config.push.enable

      push_xy = jp.where(
          push_step >= state.mjxenv_info["push_interval_steps"],
          push_xy,
          0)
      push_step = jp.where(
          push_step >= state.mjxenv_info["push_interval_steps"],
          0,
          push_step
      )
      state.mjxenv_info["push_step"] = push_step
      state.mjxenv_info["push_xy"] = push_xy
      # state.mjxenv_info["push_step"] += 1

      # print(f'step() ---> sampled push_xy: {push_xy}, push_magnitude: {push_magnitude}')
      # TODO: add push to xfrc_applied : user-defined forces in joint or Cartesian coordinates
      #  (which are stored in mjData.qfrc_applied and mjData.xfrc_applied respectively).
      # mjData.xfrc_applied are Cartesian wrenches applied to the CoM of individual bodies.
      # This field is used for example, by the native viewer to apply mouse perturbations.
      qvel = state.data.qvel
      qvel = qvel.at[:2].set(push_xy * push_magnitude + qvel[:2])
      data = state.data.replace(qvel=qvel)
      state = state.replace(data=data)
      return state

  def _handle_contact(self, state:State)->Tuple[State, jax.Array, jax.Array, jax.Array]:
      # contact = jp.array([
      #     state.data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
      #     for sensorid in self._feet_floor_found_sensor
      # ])

      # kenneth: contact sensor mode: reduce="mindist" num="1" data="found", so read value is number of contacts(points).
      # can be 0 (no contcat), 1, 2, 3, 4..
      floor_feet_contact = state.data.sensordata[self._sensor_adr.floor_feet_found_sensor_adr] > 0

      # TODO: feet_air_time accumulate the air-time from prev un-contact step un-till curr step of individual foot.
      # add ctrl_dt if not consecutive contact, and will clear feet_air_time if curr step no foot contact with floor, before exit from step().
      # case 4:
      consecutive_contact = (state.mjxenv_info["last_contact"] * floor_feet_contact)

      # add ctrl_dt when not case 4.
      state.mjxenv_info["feet_air_time"] += (self.ctrl_dt * ~consecutive_contact)

      # contact_filt = floor_feet_contact | state.mjxenv_info["last_contact"]

      # state.mjxenv_info["feet_air_time"] is 0 if prev step the corresponding foot not contact with floor.
      # state.mjxenv_info["feet_air_time"] > 0 only if prev step the corresponding foot have contact with floor.
      # first_contact = (state.mjxenv_info["feet_air_time"] > 0.0) * contact_filt

      # kenneth:last step no contact and curr step has contact, then it is `first contact` .
      # even this is the first step after reset, state.mjxenv_info["last_contact"] is True, so first_contact is False here.
      # case 2.
      first_contact = ~state.mjxenv_info["last_contact"] * floor_feet_contact

      # # TODO: feet_air_time accumulate the air-time from prev un-contact step un-till curr step of individual foot.
      # # will clear feet_air_time if curr step no foot contact with floor, at following.
      # state.mjxenv_info["feet_air_time"] += self.ctrl_dt

      accumulate_air_time = state.mjxenv_info["feet_air_time"]

      # clear running accumulated info['feet_air_time'] if contact with floor.
      state.mjxenv_info["feet_air_time"] *= ~floor_feet_contact
      state.mjxenv_info["last_contact"] = floor_feet_contact

      return state, floor_feet_contact, first_contact, accumulate_air_time

  # kenneth: collect swing peak only when foot in the air.
  def _update_swing_peak(self, state:State, floor_feet_contact: jax.Array)->Tuple[State, jax.Array]:
      # xpos in world coordinate.
      p_f = state.data.site_xpos[self._site_id.feet_site_id]
      p_fz = p_f[..., -1]
      state.mjxenv_info["swing_peak"] = jp.maximum(state.mjxenv_info["swing_peak"], p_fz)
      swing_peak_in_air = state.mjxenv_info["swing_peak"]

      # clear running statistics.
      state.mjxenv_info["swing_peak"] *= ~floor_feet_contact

      return state, swing_peak_in_air

  def _apply_jax_step(self, state:State, action:jax.Array)->Tuple[State, jax.Array, jax.Array]:
      # TODO: clip motor targets according to soft_lower/upper ? or normalize the motor_target into [lower, upper] ?
      # NOTE: action is relative to default_pose which is from keyframe `knees_bent`.
      # i.e. normalize default_pose as 0. mean of action is output of tanh, should be in [-1, 1].
      # motor_targets = self._default_pose + action * self._config.model.action_scale  # *0.5
      motor_targets = self._init_q[7:] + action * self._config.model.action_scale  # *0.5

      new_data = self.jax_step(
          # ctrl_dt=0.02, sim_dt=0.002, so n_substeps is 10.
          self.mjx_model, state.data, motor_targets, self.n_substeps
      )
      state=state.replace(data=new_data)

      last_last_act = state.mjxenv_info["last_last_act"]
      last_act = state.mjxenv_info["last_act"]

      # update
      state.mjxenv_info["motor_targets"] = motor_targets
      state.mjxenv_info["last_last_act"] = state.mjxenv_info["last_act"]
      state.mjxenv_info["last_act"] = action

      return state, last_last_act, last_act

  @staticmethod
  def _update_phase(state:State)->Tuple[State, jax.Array, jax.Array]:
      phase_tp1 = state.mjxenv_info["phase"] + state.mjxenv_info["phase_dt"]

      # kenneth: obs will use updated phase to calc cos/sin, reward use last phase.
      last_phase = state.mjxenv_info["phase"]

      # kenneth: map phase from [ 0, 2pi] -> [-pi, pi].
      state.mjxenv_info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

      # NOTE(kevin): Enable this to make the policy stand still at 0 command.
      # state.mjxenv_info["phase"] = jp.where(
      #     jp.linalg.norm(state.mjxenv_info["command"]) > 0.01,
      #     state.mjxenv_info["phase"],
      #     jp.ones(2) * jp.pi,
      # )
      new_phase = state.mjxenv_info["phase"]

      return state, last_phase, new_phase


  def _update_cmd(self, state:State,
                  rng:jax.Array
                  )->Tuple[State, jax.Array,jax.Array]:

      # disable resample command, e.g., during evaluation after training finish.
      if not self._config.command.resample_enable:
          return state, state.mjxenv_info["command"], state.mjxenv_info["command"]

      last_cmd = state.mjxenv_info["command"]

      # state.mjxenv_info["command"] = jp.where(
      #     state.mjxenv_info["step"] > 500,
      #     self.sample_command(rng),
      #     state.mjxenv_info["command"],
      # )
      # kenneth: we re-sample command after done, cause in AutoResetWrapper, will not re-sample
      # command after `done`.
      # TODO: after done, even we re-sample command here,  AutoResetWrapper will not use
      # the obs we returned, instead, it will use the reset_obs which include the command
      # from env first reset.
      # BUG: so, the obs and info["command"] is not aligned....
      #

      curr_step = state.mjxenv_info["resample_cmd_steps"]
      curr_step += 1

      state.mjxenv_info["command"] = jp.where(
              # state.done | curr_step > 500,
              curr_step >= self._config.command.resample_length, #500,
              self.sample_command(rng),
              state.mjxenv_info["command"],
      )
      new_cmd = state.mjxenv_info["command"]

      # reset if beyond 500.
      # TODO: NOTE, we don't clear resample_cmd_steps after done, cause in the beginning
      # of training, a lot of `done` happen, we don't need to do too much re-sample.
      # always re-sample according to simu-steps.
      state.mjxenv_info["resample_cmd_steps"] = jp.where(
          # state.done | (curr_step > 500)
          curr_step >= self._config.command.resample_length, # 500,
          0,
          curr_step,
      )

      return state, last_cmd, new_cmd

  # @staticmethod
  # def _update_step_count(state:State)->Tuple[State, jax.Array]:
  #     state.mjxenv_info["resample_cmd_steps"] += 1
  #     curr_step = state.mjxenv_info["resample_cmd_steps"]
  #
  #     # kenneth: NOTE: in EpisodeWrapper/AutoResetWrapper, info use `steps`, no same as here "step".
  #     state.mjxenv_info["resample_cmd_steps"] = jp.where(
  #         state.done | (state.mjxenv_info["resample_cmd_steps"] > 500),
  #         0,
  #         state.mjxenv_info["resample_cmd_steps"],
  #     )
  #     return state, curr_step


  # update new_cmd into reset_obs. for AutoResetWrapper.
  @staticmethod
  def _update_reset_obs(state:State, new_cmd:jax.Array, cmd_idx:Dict[str,slice])->State:
      assert new_cmd.shape == (3,)
      # first obs after reset.
      # first_obs:Observation = state.mjxenv_info["first_obs"]
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
      state.reset_mjxenv_info.update({
          # always deliver new rng after reset.
          "rng": state.mjxenv_info["rng"],
          "command": new_cmd,
          "resample_cmd_steps": state.mjxenv_info["resample_cmd_steps"],

          # TODO: re-sample phase_dt?
          # Phase related.
          # "phase_dt": phase_dt,
          # after done, the data is reset to the beginning status, so phase should be back to [0,pi],
          # no need to update new_phase to reset_info.
          # and in reset_obs, the phase is cos/sin of [0,pi].
          # "phase": phase,  # [0, pi]

          # TODO: re-sample Push ?
          # keep the push_step count, so we can get more random sampling on push_xy.
          "push_xy": state.mjxenv_info["push_xy"],
          "push_step": state.mjxenv_info["push_step"],
          "push_interval_steps": state.mjxenv_info["push_interval_steps"],

          # record the data/obs after reset, then used when step() `done` through soft-reset mechanism.
          # 'first_data': first_data,
          # NOTE: in first_obs, the  "command" maybe different as step() used when `done`, cause the
          # command will be re-sampled every 500-steps in step().
          # 'first_obs': first_obs,
      })

      # TODO: we can not replace state.info which also include the outter-wrapper specific key/values.
      # state=state.replace(reset_info=reset_info)

      return state


  # metrics can be used by EvalWrapper, during the validation(eval) per training epoch.
  # and progress_fn can write interested metrics to TensorboardX for plotting.
  # TODO : add more metrics data for plotting in TensorBoardX.
  @staticmethod
  def _update_metrics(state:State, rewards: Dict[str, jax.Array], swing_peak_in_air:jax.Array)->State:
      for k, v in rewards.items():
          state.metrics[f"reward/{k}"] = v

      # kenneth
      # state.metrics["swing_peak"] = jp.mean(state.mjxenv_info["swing_peak"])
      state.metrics["swing_peak"] = jp.mean(swing_peak_in_air)

      return state


  # --- jit boundary ---
  # TODO:NOTE: we can not replace state.info which also include the outter-wrapper specific key/values.
  def step(self, state: State, action: jax.Array) -> State:
     # after we update rng, we will update new rng into reset_info also.
     state.mjxenv_info["rng"], key_push, key_obs, key_cmd = jax.random.split(state.mjxenv_info["rng"], 4)

    # kenneth: flax.struct.dataclass is a 'frozen' dataclass, so must use .replace() to modify member.
    # note: .replace() do shallow copy on other non_replaced data members, so it is efficient
    # to do .replace() multiple times in each child functions here, especially used in jit.

    # add push to qvel.
     state = self._apply_push(state, key_push)

     # TODO: for evaluate, use mujoco on CPU can be faster than mjx which is suitable for multiple-env-instances.
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
     # TODO: if done is caused by nan, we stop calc rewards...
     rewards = self._rwd_handler.get_rewards(
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
         soft_joint_range=self._soft_joint_range,
         init_q=self._init_q,
         cost_pose_weight=self._cost_pose_weight,
     )
     # TODO: kenneth: handle rewards get nan:
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
     # output_data = jp.where(state.done, state.mjxenv_info["first_data"], state.data)
     # state = state.replace(data=output_data)

    # state.mjxenv_info["step"] += 1
    # state.mjxenv_info["push_xy"] = push_xy
    # state.mjxenv_info["push_step"] += 1
    # state.mjxenv_info["motor_targets"] = motor_targets

    # phase_tp1 = state.mjxenv_info["phase"] + state.mjxenv_info["phase_dt"]
    #
    # # kenneth: map phase from [ 0, 2pi] -> [-pi, pi].
    # state.mjxenv_info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # # state.mjxenv_info["phase"] = jp.where(
    # #     jp.linalg.norm(state.mjxenv_info["command"]) > 0.01,
    # #     state.mjxenv_info["phase"],
    # #     jp.ones(2) * jp.pi,
    # # )

    # state.mjxenv_info["last_last_act"] = state.mjxenv_info["last_act"]
    # state.mjxenv_info["last_act"] = action

    # state.mjxenv_info["rng"], cmd_rng = jax.random.split(state.mjxenv_info["rng"])
    # state.mjxenv_info["command"] = jp.where(
    #     state.mjxenv_info["step"] > 500,
    #     self.sample_command(cmd_rng),
    #     state.mjxenv_info["command"],
    # )
    # state.mjxenv_info["step"] = jp.where(
    #     done | (state.mjxenv_info["step"] > 500),
    #     0,
    #     state.mjxenv_info["step"],
    # )

    # TODO:   merge with state.mjxenv_info["feet_air_time"] += self.ctrl_dt....
    # e.g. state.mjxenv_info["feet_air_time"] =+  self.ctrl_dt * ~contact..
    #  no, must set state.mjxenv_info["feet_air_time"] to zero as sentinal for next step...
    # state.mjxenv_info["feet_air_time"] *= ~contact
    # state.mjxenv_info["last_contact"] = contact
    # state.mjxenv_info["swing_peak"] *= ~contact

    # for k, v in rewards.items():
    #   state.metrics[f"reward/{k}"] = v
    #
    # # kenneth
    # # state.metrics["swing_peak"] = jp.mean(state.mjxenv_info["swing_peak"])
    # state.metrics["swing_peak"] = jp.mean(swing_peak_in_air)

    # done = done.astype(reward.dtype)
    # state = state.replace(data=data, obs=obs, reward=reward, done=done)
    # state = state.replace(obs=obs, reward=reward, done=done)
     return state


  def _update_termination(self, state: State) -> State:
    # z axis should be along world coordinate.
    # TODO: should not allow some large tilt angle.
    fall_termination = state.data.sensordata[self._sensor_adr.pelvis_upvector_sensor_adr][-1] < 0.0

    contact_termination = jp.any(state.data.sensordata[ self._sensor_adr.left_leg_right_leg_found_sensor_adr] > 0)

    done = (
            fall_termination
            | contact_termination
            | jp.isnan(state.data.qpos).any()
            | jp.isnan(state.data.qvel).any()
    #         TODO: kenneth: add more judgement on nan:
            | jp.isnan(state.data.qacc).any()
            | jp.isnan(state.data.sensordata).any()
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
      gyro = data.sensordata[self._sensor_adr.pelvis_gyro_sensor_adr]
      noisy_gyro = (
              gyro
              + (2 * jax.random.uniform(key_gyro, shape=gyro.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.gyro
      )

      # convert gravity unit vector from global to imu_site local frame.
      gravity = data.site_xmat[self._site_id.pelvis_imu_site_id].T @ jp.array([0, 0, -1])
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
      phase_cos_sin = jp.concatenate([jp.cos(phase), jp.sin(phase)])

      # linvel = self.get_local_linvel(data, "pelvis")
      linvel = data.sensordata[self._sensor_adr.pelvis_local_linvel_sensor_adr]
      # info["rng"], noise_rng = jax.random.split(info["rng"])
      noisy_linvel = (
              linvel
              + (2 * jax.random.uniform(key_linvel, shape=linvel.shape) - 1)
              * self._config.model.noise.level
              * self._config.model.noise.scales.linvel
      )

      # TODO: check all obs values can be got on real robot through sensors.
      # the obs after soft-reset, must be aligned with the info which also contain
      # several same elements as obs, e.g., cmd.
      # shape: (56,)
      policy_state = jp.hstack([
          # # info["command"], # 3
          cmd,  # 3
          noisy_linvel,  # 3
          noisy_gyro,  # 3
          noisy_gravity,  # 3
          # # # info["command"], # 3
          # cmd,               # 3
          noisy_joint_angles - self._init_q[7:], #self._default_pose,  # 29
          noisy_joint_vel,  # 29
          # kenneth: we should record curr_act and last_act in obs for next step.
          # info["last_act"],  # 29
          curr_act,  # 29
          last_act,  # 29
          phase_cos_sin, # 4
      ])

      # accelerometer = self.get_accelerometer(data, "pelvis")
      # global_angvel = self.get_global_angvel(data, "pelvis")

      accelerometer = data.sensordata[self._sensor_adr.pelvis_accelerometer_sensor_adr]
      global_angvel = data.sensordata[self._sensor_adr.pelvis_global_angvel_sensor_adr]

      feet_vel = data.sensordata[self._sensor_adr.feet_linvel_sensor_adr].ravel()
      root_height = data.qpos[2]

      # shape: (112,)
      privileged_state = jp.hstack([
          policy_state,
          gyro,  # 3
          accelerometer,  # 3
          gravity,  # 3
          linvel,  # 3
          global_angvel,  # 3
          joint_angles - self._init_q[7:],  # self._default_pose,
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


  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.command.lin_vel_x[0], maxval=self._config.command.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.command.lin_vel_y[0], maxval=self._config.command.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.command.ang_vel_yaw[0],
        maxval=self._config.command.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

if __name__ == "__main__":
    test_task = 'flat_terrain'
    test_env = Joystick(task=test_task)
    rng=jax.random.key(0)
    reset_state = test_env.reset(rng)
    # print(f'{reset_state.data.qpos=:}')

    assert reset_state.data == reset_state.reset_data
    assert reset_state.mjxenv_info == reset_state.reset_mjxenv_info
    assert reset_state.obs == reset_state.reset_obs

    def _check_nan(x:jax.Array):
        has_nan = jp.any(jp.isnan(x))
        assert not has_nan
        return has_nan

    print(jax.tree.map(_check_nan, reset_state))





