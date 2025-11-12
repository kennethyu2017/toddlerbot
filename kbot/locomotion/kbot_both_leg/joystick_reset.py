from typing import Dict, Tuple, Sequence
import jax
import jax.numpy as jp
from jax.typing import ArrayLike
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from kbot.base_env.base_env_mjx import make_mjx_data

class JoystickResetHelper:

	@staticmethod
	def _rand_qpos(qpos: jax.Array,
				   rng: jax.Array,
				   soft_lowers:ArrayLike,
				   soft_uppers:ArrayLike)->jax.Array:
		rng, key_free_xy, key_free_yaw, key_actuator_qpos = jax.random.split(rng, 4)

		# randomize free joint pos:
		# x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).

		dxy = jax.random.uniform(key_free_xy, (2,), minval=-0.5, maxval=0.5)
		qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

		yaw = jax.random.uniform(key_free_yaw, (1,), minval=-3.14, maxval=3.14)
		quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
		new_quat = math.quat_mul(qpos[3:7], quat)
		qpos = qpos.at[3:7].set(new_quat)

		# TODO: qpos is not normalized, * 0.5, * 1.5 not stable? clip to range?
		# randomize joint qpos:
		# qpos[7:]=*U(0.5, 1.5)
		# qpos = qpos.at[7:].set(
		#     # qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
		#     qpos[7:] * jax.random.uniform(key, (self._mj_model.nq - 7,), minval=0.5, maxval=1.5)
		# )
		# new_jnt_qpos = qpos[7:] * jax.random.uniform(key, (self._mj_model.nq - 7,), minval=0.5, maxval=1.5)
		new_jnt_qpos = qpos[7:] * jax.random.uniform(key_actuator_qpos, qpos[7:].shape, minval=0.5, maxval=1.5)
		new_jnt_qpos = jp.clip(new_jnt_qpos, min=soft_lowers, max=soft_uppers)
		qpos = qpos.at[7:].set(new_jnt_qpos)
		return qpos

	@staticmethod
	def _rand_qvel(qvel: jax.Array, rng: jax.Array) -> jax.Array:
		# TODO: set qvel of dof joints.
		# freejoint linve and angvel.
		# d(xyzrpy)=U(-0.5, 0.5)
		rng, key_qvel = jax.random.split(rng)
		qvel = qvel.at[0:6].set(
			jax.random.uniform(key_qvel, (6,), minval=-0.5, maxval=0.5)
		)
		return qvel

	@staticmethod
	def _rand_phase_dt(rng: jax.Array, ctrl_dt:float) -> Tuple[jax.Array, jax.Array]:
		# Phase, freq=U(1.0, 1.5)
		# finish gait_freq*2pi per second.
		rng, key_phase = jax.random.split(rng)
		gait_freq = jax.random.uniform(key_phase, (1,), minval=1.25, maxval=1.5)
		phase_dt = 2 * jp.pi * ctrl_dt * gait_freq
		phase = jp.array([0, jp.pi])
		return phase_dt, phase

	@staticmethod
	def _rand_push(rng: jax.Array, ctrl_dt:float, lower:ArrayLike, upper:ArrayLike) -> jax.Array:
		# rng, cmd_rng = jax.random.split(rng)
		# cmd = self.sample_command(cmd_rng)

		# Sample push interval.
		rng, key_push = jax.random.split(rng)
		# in second.
		push_interval = jax.random.uniform(
			key_push,
			minval=lower, #  self._config.push_config.interval_range[0],
			maxval=upper, #  self._config.push_config.interval_range[1],
		)
		push_interval_steps = jp.round(push_interval / ctrl_dt).astype(jp.int32)
		return push_interval_steps

	@staticmethod
	def gen_data(*, mj_model:mujoco.MjModel,
				  mjx_model: mjx.Model,
				  init_qpos:ArrayLike,
				  init_qvel:ArrayLike,
				  soft_lowers:ArrayLike,
				  soft_uppers:ArrayLike,
				  rng:jax.Array)-> mjx.Data:
		qpos, qvel = init_qpos, init_qvel
		rng, key_qpos, key_qvel = jax.random.split(rng, 3)

		print(f'reset() ---> init qpos before randomization: {qpos=:} init qvel: {qvel=:}')
		qpos= JoystickResetHelper._rand_qpos(qpos, key_qpos, soft_lowers, soft_uppers)
		# print(f'qpos after randomization: {qpos=:}')

		qvel = JoystickResetHelper._rand_qvel(qvel, key_qvel)
		# print(f'qvel after randomization: {qvel=:}')

		data = make_mjx_data(
			mj_model,
			qpos=qpos,
			qvel=qvel,
			ctrl=qpos[7:],
			impl=mjx_model.impl.value,
			# todo: #nconmax,njmax are deprecated only use for mujoco prior to 2.3.0.
			# and only used for `warp` backend.
			# nconmax=self._config.nconmax,
			# njmax=self._config.njmax,
		)
		# kenneth: important to do FK to make mjc stable.
		data = mjx.forward(mjx_model, data)
		return data

	@staticmethod
	def gen_info(*,
				  # record the data/obs after reset, then used when step() `done` through soft-reset mechanism.
				  # first_data: mjx.Data,
				  # first_obs: Observation,
				  rng:jax.Array,
				  ctrl_dt:float,
				  push_interval_lower:ArrayLike,
				  push_interval_upper:ArrayLike,
				  cmd:ArrayLike,
				  nu:int)->Dict[str, jax.Array]:
		rng, key_phase, key_push, key_info = jax.random.split(rng, 4)

		phase_dt, phase = JoystickResetHelper._rand_phase_dt(key_phase, ctrl_dt)
		# print(f'phase_dt after randomization: {phase_dt=:} {phase=:}')

		push_interval_steps = JoystickResetHelper._rand_push(key_push,
															 ctrl_dt,
															 push_interval_lower,
															 push_interval_upper)
		# print(f'push_interval_steps after randomization: {push_interval_steps=:}')

		# note: all leaf nodes must be jax.Array type to be able to cross jit boundary.
		mjxenv_info = {
			"rng": key_info,  # rng,
			"resample_cmd_steps": 0,
			"command": cmd,
			"last_act": jp.zeros(nu),
			"last_last_act": jp.zeros(nu),
			"motor_targets": jp.zeros(nu),
			"feet_air_time": jp.zeros(2),  # air time of individual left/right feet.

			# kenneth: after reset the feet of robot should be on floor, causing we
			# do FK through mjx.forward() in gen_data.
			# and we use fine-tuned keyframe `knees_bent` to guarantee the
			# feet on floor after reset (viewed in mujoco.viewer).
			# "last_contact": jp.zeros(2, dtype=bool),
			"last_contact": jp.ones(2, dtype=bool),

			"swing_peak": jp.zeros(2),  # left/right feet

			# Phase related.
			"phase_dt": phase_dt,
			"phase": phase,   # [0, pi]

			# Push related.
			"push_xy": jp.array([0.0, 0.0]),
			"push_step": 0,
			"push_interval_steps": push_interval_steps,

			# record the data/obs after reset, then used when step() `done` through soft-reset mechanism.
			# 'first_data': first_data,
			# NOTE: in first_obs, the  "command" maybe different as step() used when `done`, cause the
			# command will be re-sampled every 500-steps in step().
			# 'first_obs': first_obs,
		}
		return mjxenv_info


	@staticmethod
	def gen_metrics(reward_keys: Sequence[str])->Dict[str, jax.Array]:
		metrics = {}
		# for k in self._config.reward.scales.keys():
		for k in reward_keys:
			metrics[f"reward/{k}"] = jp.zeros(())
		metrics["swing_peak"] = jp.zeros(())
		return metrics

	#
	# @staticmethod
	# def _gen_obs()->Observation:
	# 	contact = jp.array([
	# 		data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
	# 		for sensorid in self._feet_floor_found_sensor
	# 	])
	# 	obs = self._get_obs(data, info, contact)
	# 	return obs
	#