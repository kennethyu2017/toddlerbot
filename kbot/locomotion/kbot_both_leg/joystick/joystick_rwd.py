
from typing import Dict, List
import jax
import jax.numpy as jp
import mujoco.mjx as mjx
from mujoco_playground._src import gait

from ml_collections import config_dict

class JoystickReward:
	def __init__(self, *,
				 joint_adr: config_dict.FrozenConfigDict,
				 sensor_adr: config_dict.FrozenConfigDict,
				 site_id:config_dict.FrozenConfigDict,
				 rwd_cfg: config_dict.FrozenConfigDict) -> None:
		# include freejoint.
		self._joint_adr = joint_adr
		self._sensor_adr = sensor_adr
		self._site_id = site_id
		self._rwd_cfg = rwd_cfg

	def get_rewards(
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
			last_phase: jax.Array,
			soft_joint_range: List[jax.Array],
			# including freejoint.
			init_q: jax.Array,
	) -> Dict[str, jax.Array]:
		# del metrics  # Unused.
		return {
			# Tracking rewards.
			"tracking_lin_vel": self._reward_tracking_lin_vel(
				# info["command"], self.get_local_linvel(data, "pelvis")
				last_cmd,
				# self.get_local_linvel(data, "pelvis")
				data.sensordata[self._sensor_adr.pelvis_local_linvel_sensor_adr]
			),
			"tracking_ang_vel": self._reward_tracking_ang_vel(
				# info["command"], self.get_gyro(data, "pelvis")
				last_cmd,
				# self.get_gyro(data, "pelvis")
				data.sensordata[self._sensor_adr.pelvis_gyro_sensor_adr]
			),
			# Base-related rewards.
			"lin_vel_z": self._cost_lin_vel_z(
				# self.get_global_linvel(data, "pelvis"),
				# self.get_global_linvel(data, "torso"),
				data.sensordata[self._sensor_adr.pelvis_global_linvel_sensor_adr],
			),
			"ang_vel_xy": self._cost_ang_vel_xy(
				# self.get_global_angvel(data, "torso")
				data.sensordata[self._sensor_adr.pelvis_global_angvel_sensor_adr],
			),

			"orientation": self._cost_orientation(
				# self.get_gravity(data, "torso")
				data.sensordata[self._sensor_adr.pelvis_upvector_sensor_adr],
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
				data, last_phase, self._rwd_cfg.max_foot_height, last_cmd
			),
			# Other rewards.
			"alive": self._reward_alive(),
			"termination": self._cost_termination(done),
			# "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
			# "stand_still": self._cost_stand_still(last_cmd, data.qpos[7:]),
			"stand_still": self._cost_stand_still(data, last_cmd, init_q),
			"hand_collision": self._cost_hand_collision(data),
			"contact_force": self._cost_contact_force(data),
			# Pose related rewards.
			"joint_deviation_hip": self._cost_joint_deviation_hip(
				# data.qpos[7:], info["command"]
				data, last_cmd, init_q
			),
			# "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
			"joint_deviation_knee": self._cost_joint_deviation_knee(data, last_cmd, init_q),
			# "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
			"dof_pos_limits": self._cost_joint_pos_limits(data,soft_joint_range),
			# "pose": self._cost_pose(data.qpos[7:]),
			"pose": self._cost_pose(data,init_q),
		}

	def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
		# l_contact_force = mjx_env.get_sensor_data(
		#     self.mj_model, data, "left_foot_force"
		# )

		# both feet, ndim=3 contact frc.
		feet_contact_frc = data.sensordata[self._sensor_adr.feet_force_sensor_adr]

		# r_contact_force = mjx_env.get_sensor_data(
		#     self.mj_model, data, "right_foot_force"
		# )

		l_z_frc, r_z_frc = feet_contact_frc[jp.array([2, 5])]

		# only penalty on contac_force > 500, i.e. jump and fall onto floor.
		cost = jp.clip(
			# jp.abs(l_contact_force[2])
			jp.abs(l_z_frc)
			- self._rwd_cfg.max_contact_force,
			min=0.0,
		)
		cost += jp.clip(
			# jp.abs(r_contact_force[2])
			jp.abs(r_z_frc)
			- self._rwd_cfg.max_contact_force,
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
			self, data: mjx.Data, cmd: jax.Array, init_q:jax.Array
	) -> jax.Array:
		# hip roll (l,r), hip yaw (l,r) :
		# error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
		error = data.qpos[self._joint_adr.hip_r_y_jnt_adr] - init_q[self._joint_adr.hip_r_y_jnt_adr]

		# Allow roll deviation when lateral velocity is high.
		# weight = jp.where(
		#     # cmd[1] > 0.1,
		#     jp.abs(cmd[1]) > 0.05,
		#     # left - r y, right - r y,
		#     # jp.array([0.0, 1.0, 0.0, 1.0]),
		#
		#     # hip roll (l,r), hip yaw (l,r) :
		#     jp.array([0.0, 0.0, 1.0, 1.0]),
		#     jp.array([1.0, 1.0, 1.0, 1.0]),
		# )
		# # kenneth:
		# # Allow yaw deviation when ang_yaw velocity is high.
		# weight *= jp.where(
		#     jp.abs(cmd[2]) > 0.1,
		#     # left - r y, right - r y,
		#     # jp.array([0.0, 1.0, 0.0, 1.0]),
		#
		#     # hip roll (l,r), hip yaw (l,r) :
		#     jp.array([1.0, 1.0, .0, .0]),
		#     jp.array([1.0, 1.0, 1.0, 1.0]),
		# )

		# kenneth: kbot does not have waist, during waling , we allow more freedom on hips.
		weight = jp.where(
			# cmd[1] > 0.1,
			jp.linalg.norm(cmd) > 0.01,
			# left - r y, right - r y,
			# jp.array([0.0, 1.0, 0.0, 1.0]),

			# hip roll (l,r), hip yaw (l,r) :
			jp.array([0.0, 0.0, 0.0, 0.0]),
			jp.array([1.0, 1.0, 1.0, 1.0]),
		)

		cost = jp.sum(jp.abs(error) * weight)
		return cost

	def _cost_joint_deviation_knee(self, data: mjx.Data, cmd:jax.Array, init_q: jax.Array) -> jax.Array:
		# error = qpos[self._knee_p_jnt_adr] - self._default_pose[self._knee_p_jnt_adr]
		# init_q including free joint.
		error = data.qpos[self._joint_adr.knee_p_jnt_adr] - init_q[self._joint_adr.knee_p_jnt_adr]
		# return jp.sum(jp.abs(error))
		weight = jp.where(
			jp.linalg.norm(cmd) > 0.01,
			0.0,
			1.0,
		)
		return weight * jp.sum(jp.abs(error))

	def _cost_pose(self, data: mjx.Data, init_q: jax.Array) -> jax.Array:
		# return jp.sum(jp.square(qpos - self._default_pose))
		# return jp.sum(jp.square(data.qpos[7:] - self._default_pose))
		# init_q including the free joint.
		return jp.sum(jp.square(data.qpos[7:] - init_q[7:]))

	# exclude the freejnt
	def _cost_joint_pos_limits(self, data: mjx.Data, soft_joint_range:List[jax.Array]) -> jax.Array:
		# assert np.all(soft_joint_range[0] < soft_joint_range[1])
		out_of_limits = -jp.clip(data.qpos[7:] - soft_joint_range[0], None, 0.0)
		out_of_limits += jp.clip(data.qpos[7:] - soft_joint_range[1], 0.0, None)
		return jp.sum(out_of_limits)

	@staticmethod
	def _projection_a_onto_b(a: jax.Array, b: jax.Array) -> jax.Array:
		"""
		Projects vector a onto vector b.

		Args:
			a (np.array): The vector to be projected.
			b (np.array): The vector to project onto.

		Returns:
			np.array: The vector projection of a onto b.
		"""
		# Calculate the dot product of vector a and vector b
		dot_product = jp.dot(a, b)

		# Calculate the dot product of vector b with itself (magnitude squared)
		b_squared = jp.dot(b, b)

		# If the squared magnitude is zero, vector b is a zero vector,
		# so the projection is also the zero vector.

		# Calculate the scalar component of the projection
		# scalar_component = jp.where(
		#     jp.any(b_squared == 0),
		#     jp.zeros_like(a),
		#     dot_product / (b_squared + 1e-6),
		# )
		scalar_component = dot_product / (b_squared + 1e-6)

		# Multiply the scalar component by vector b to get the projected vector
		projection = scalar_component * b
		return projection

	# def _reward_tracking_lin_vel(
	#     self,
	#     command: jax.Array,
	#     local_vel: jax.Array,
	# ) -> jax.Array:
	#   #   kenneth: TODO: should use relative error, because command is changing.
	#   # lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
	#   # return jp.exp(-lin_vel_error / self._config.reward.tracking_sigma)
	#
	#   # kenneth: use 2D dot product to calc tracking.
	#   # if opposite direction, dot result < 0 as penalty.
	#   # return 0.1 * commands[:2].dot(local_vel[:2])
	#   def _signed_proj_clipped(cmd, vel):
	#       vector_cmd = jp.array([0., cmd])
	#       vector_vel = jp.array([0., vel])
	#       projection = self._projection_a_onto_b(vector_vel, vector_cmd)
	#       # project has format: [0., signed_proj_value], so we can sum together to get signed norm of projection.
	#       # bound to [-1,1] times of abs(cmd).
	#       # TODO: in case cmd is zero.
	#       projection=jp.where(
	#           jp.abs(cmd) == 0.,
	#           # -1 to penalty.
	#           (jp.abs(vel) != 0) * -1,
	#           projection,
	#       )
	#       # bound to [-1,1] times of abs(cmd).
	#       return jp.clip(
	#           jp.sum(projection) / ( jp.abs(cmd) + 1e-6 ),
	#           min=-1.,
	#           max=1.)
	#
	#   # signed cost.
	#   x_cost = _signed_proj_clipped(command[0], local_vel[0])
	#   # y_cost = _signed_proj_clipped(command[1], local_vel[1])
	#   return x_cost # + y_cost

	def _reward_tracking_lin_vel(
			self,
			command: jax.Array,
			local_vel: jax.Array,
	) -> jax.Array:
		# kenneth: TODO: should use relative error, because command is changing.
		lin_vel_error = jp.sum(jp.square(command[:2] - local_vel[:2]))
		return jp.exp(-lin_vel_error / self._rwd_cfg.tracking_sigma)

	# tracking yaw.
	# def _reward_tracking_ang_vel(
	#     self,
	#     command: jax.Array,
	#     ang_vel: jax.Array,
	# ) -> jax.Array:
	#   # ang_vel_error = jp.square(commands[2] - ang_vel[2])
	#   # return jp.exp(-ang_vel_error / self._config.reward.tracking_sigma)
	#
	#   # kenneth: construct 2D dot product to calc yaw tracking. negative if opposite direction.
	#   # return 0.1 * jp.array([0., commands[2]]).dot(jp.array([0., ang_vel[2]]) )
	#   def _signed_proj_clipped(cmd, vel):
	#       vector_cmd = jp.array([0., cmd])
	#       vector_vel = jp.array([0., vel])
	#       projection = self._projection_a_onto_b(vector_vel, vector_cmd)
	#       # project has format: [0., signed_proj_value], so we can sum together to get signed norm of projection.
	#       # bound to [-1,1] times of abs(cmd).
	#       # TODO: in case cmd is zero.
	#       projection=jp.where(
	#           jp.abs(cmd) == 0.,
	#           # -1 to penalty.
	#           (jp.abs(vel) != 0) * -1,
	#           projection,
	#       )
	#       # bound to [-1,1] times of abs(cmd).
	#       return jp.clip(
	#           jp.sum(projection) / ( jp.abs(cmd) + 1e-6 ),
	#           min=-1.,
	#           max=1.)
	#
	#   # signed cost.
	#   yaw_cost = _signed_proj_clipped(command[2], ang_vel[2])
	#   return yaw_cost

	def _reward_tracking_ang_vel(
			self,
			command: jax.Array,
			ang_vel: jax.Array,
	) -> jax.Array:
		ang_vel_error = jp.square(command[2] - ang_vel[2])
		return jp.exp(-ang_vel_error / self._rwd_cfg.tracking_sigma)

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
		# TODO: g1 use jp.array([0.073, 0.0, 1.0]), read from sensordata: framezaxis "upvector_torso"
		# after load keyframe 'knees_bent'.
		# return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

		# framezaxis returns the 3D unit vector corresponding to the Z-axis of
		# the spatial frame of the object, in global coordinates, so
		# we can subtract the two unit-vector to get the orientation error.
		# jp.array([0., 0., 1.0]) is read from sensordata: framezaxis "upvector_torso"
		# after load keyframe 'knees_bent', the default pose.
		return jp.sum(jp.square(pelvis_zaxis - jp.array([0., 0., 1.0])))

	def _cost_base_height(self, data: mjx.Data) -> jax.Array:
		return jp.square(
			# freejoint z.
			data.qpos[2] - self._rwd_cfg.base_height_target
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
			self, data: mjx.Data, commands: jax.Array, init_q: jax.Array
	) -> jax.Array:
		cmd_norm = jp.linalg.norm(commands)
		# cost = jp.sum(jp.abs(data.qpos[7:] - self._default_pose))

		# init_q including the freejoint.
		cost = jp.sum(jp.abs(data.qpos[7:] - init_q[7:]))
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
		# TODO: feet vel is too big for penalty compared against pelvis vel?

		# TODO: kenneth: penalty on feet vel instead on pelvis vel.
		# body_vel = self.get_global_linvel(data, "pelvis")[:2]
		# body_vel = data.sensordata[self._pelvis_global_linvel_sensor_adr][:2]

		feet_vel = data.sensordata[self._sensor_adr.feet_linvel_sensor_adr]

		# vel_xy = feet_vel[..., :2]
		l_vel_xy, r_vel_xy = feet_vel[:2], feet_vel[3:5]

		# kenneth: norm already include sqrt...
		# vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
		l_vel_norm = jp.linalg.norm(l_vel_xy)
		r_vel_norm = jp.linalg.norm(r_vel_xy)
		# slip: contact with floor when foot has vel_xy.
		cost = jp.sum(jp.array([l_vel_norm, r_vel_norm]) * floor_feet_contact)

		return cost

	# TODO: debug.
	def _cost_feet_clearance(
			self, data: mjx.Data
	) -> jax.Array:
		# vel in global coordinate.
		feet_vel = data.sensordata[self._sensor_adr.feet_linvel_sensor_adr]
		# TODO: bug on index [..., :2]
		vel_xy = feet_vel[..., :2]

		vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
		# feet site is left_foot_ankle/right_foot_ankle.
		foot_pos = data.site_xpos[self._site_id.feet_site_id]
		foot_z = foot_pos[..., -1]
		# max_foot_height is 0.15
		delta = jp.abs(foot_z - self._rwd_cfg.max_foot_height)
		# require keep foot ankle at max height when high vel.
		return jp.sum(delta * vel_norm)

	def _cost_feet_height(
			self,
			swing_peak: jax.Array,
			first_contact: jax.Array,
	) -> jax.Array:
		#  encourage swing peak close to max foot height.
		error = swing_peak / self._rwd_cfg.max_foot_height - 1.0
		return jp.sum(jp.square(error) * first_contact)

	def _reward_feet_air_time(
			self,
			# in sec.
			air_time: jax.Array,
			first_contact: jax.Array,
			commands: jax.Array,
			# in sec.
			threshold_min: float = 0.2,
			threshold_max: float = 0.5,
	) -> jax.Array:
		del commands  # Unused.
		# encourage feet air time > 0.2 sec
		air_time = (air_time - threshold_min) * first_contact
		air_time = jp.clip(air_time, max=threshold_max - threshold_min)
		reward = jp.sum(air_time)
		return reward

	def _reward_feet_phase(
			self,
			data: mjx.Data,
			phase: jax.Array,
			max_foot_height: jax.Array,
			command: jax.Array,
	) -> jax.Array:
		# feet site is foot_ankle.
		foot_pos = data.site_xpos[self._site_id.feet_site_id]
		foot_z = foot_pos[..., -1]
		rz = gait.get_rz(phase, swing_height=max_foot_height)
		error = jp.sum(jp.square(foot_z - rz))
		tracking_sigma = 0.01
		reward = jp.exp(-error / tracking_sigma)

		# body_linvel = self.get_global_linvel(data, "pelvis")[:2]
		# body_angvel = self.get_global_angvel(data, "pelvis")[2]

		body_linvel = data.sensordata[self._sensor_adr.pelvis_global_linvel_sensor_adr][:2]
		# around global z-axis
		body_angvel = data.sensordata[self._sensor_adr.pelvis_global_angvel_sensor_adr][2]

		body_vel_mask = jp.logical_or(
			jp.linalg.norm(body_linvel) > 0.1,
			jp.abs(body_angvel) > 0.1,
		)

		command_mask = jp.logical_or(
			jp.linalg.norm(command[:2]) > 0.01,
			jp.abs(command[2]) > 0.01,
		)

		# mask = jp.logical_or(linvel_mask, jp.linalg.norm(command) > 0.01)
		mask = jp.logical_or(body_vel_mask, command_mask)

		reward *= mask
		return reward
