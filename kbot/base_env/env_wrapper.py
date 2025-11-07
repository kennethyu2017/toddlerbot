
"""Wrappers for MuJoCo Playground environments."""

import contextlib
import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env,wrapper
import numpy as np


class SoftResetWrapper(wrapper.Wrapper):
	"""Automatically soft resets Brax envs that are done.

	`full_reset` is disabled :
	  * the environment will reset to a cached first state.
	  * only data is reset, not the environment info and obs.
	  * env info is inherited from inner_env.
	  * obs is get from inner_env for special case of  "command" which is re-sampled in MjxEnv every 500-steps, and different as
	    first state obs.

	Attributes:
	  env: The wrapped environment.

	"""

	def __init__(self, env: Any ):
		super().__init__(env)
		self._info_key = 'SoftResetWrapper'

	def reset(self, rng: jax.Array) -> mjx_env.State:
		rng_key = jax.vmap(jax.random.split)(rng)
		rng, key = rng_key[..., 0], rng_key[..., 1]

		# kenneth: state from EpisodeWrapper.reset() which add 'truncation'/'episode_done'... into state.info
		state = self.env.reset(key)

		key_shape_slice = slice(0, max(1 ,(len(key.shape) - 1)))

		# state.info[f'{self._info_key}_first_data'] = state.data
		# state.info[f'{self._info_key}_first_obs'] = state.obs

		state.info[f'{self._info_key}_done_count'] = jp.zeros(
			# key.shape[:-1], dtype=int
			key.shape[key_shape_slice], dtype=int
		)
		return state

	def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
		# grab the reset state.
		reset_state = None

		#  kenneth: BUG: the command value in reset_obs maybe different as in state.info["command"] which
		#  is inheritted from prev step state with re-sampling command every 500-steps.
		# the MjxEnv.step() can output soft_reset_obs and first_data now.
		# reset_data = state.info[f'{self._info_key}_first_data']
		# reset_obs = state.info[f'{self._info_key}_first_obs']

		# TODO: after done, even we re-sample command here,  AutoResetWrapper will not use
		# the obs we returned, instead, it will use the reset_obs which include the command
		# from env first reset.
		# BUG: so, the obs and info["command"] is not aligned....
		#

		if 'steps' in state.info:
			# reset steps to 0 if done.
			steps = state.info['steps']
			steps = jp.where(state.done, jp.zeros_like(steps), steps)
			state.info.update(steps=steps)

		state = state.replace(done=jp.zeros_like(state.done))
		state = self.env.step(state, action)

		# kenneth: inherit the state.info including  "command"/"step"/"last_contact"/"last_act" etc,
		# but will be replaced with reset_state.info if full_reset.
		# the MjxEnv.step() can output soft_reset_obs and first_data now.
		next_info = state.info

		done_count_key = f'{self._info_key}_done_count'

		# kenneth: if full_reset, will use the reset_state.info which contains the re-sampled "command",
		# and is same as in reset_obs.
		if self._full_reset and reset_state:
			next_info = jax.tree.map(where_done, reset_state.info, state.info)
			next_info[done_count_key] = state.info[done_count_key]

			if 'steps' in next_info:
				next_info['steps'] = state.info['steps']
			preserve_info_key = f'{self._info_key}_preserve_info'
			if preserve_info_key in next_info:
				next_info[preserve_info_key] = state.info[preserve_info_key]

		next_info[done_count_key] += state.done.astype(int)


		return state.replace(data=data, obs=obs, info=next_info)
