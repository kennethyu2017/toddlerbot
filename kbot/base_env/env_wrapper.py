
"""Wrappers for MuJoCo Playground environments."""


from typing import Any, Tuple
import jax
from jax import numpy as jp
from jax._src.lib import pytree
from mujoco_playground._src import wrapper
from base_env_mjx import State

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

	def reset(self, rng: jax.Array) -> State:
		rng_key = jax.vmap(jax.random.split)(rng)
		rng, key = rng_key[..., 0], rng_key[..., 1]

		# kenneth: state from EpisodeWrapper.reset() which add 'truncation'/'episode_done'... into state.info
		state = self.env.reset(key)

		key_shape_slice = slice(0, max(1, (len(key.shape) - 1)))

		# kenneth: if we record state.data/obs into self as data member, that will
		# be a side-effect of a jax-jit function (and save a Tracer object actually).
		# Record into state to make reset() as pure as possible.
		# state.info[f'{self._info_key}_first_data'] = state.data
		# state.info[f'{self._info_key}_first_obs'] = state.obs
		# state.info[f'{self._info_key}_rng'] = rng

		state.info[f'{self._info_key}_done_count'] = jp.zeros(
			# key.shape[:-1], dtype=int
			key.shape[key_shape_slice], dtype=int
		)
		return state

	def step(self, state: State, action: jax.Array)->State:
		# reset_xxx is updated by inner-most MjxEnv.step()
		# reset_data, reset_obs, reset_info only updated by the inner-most MjxEnv.
		inner_reset_data=state.reset_data
		inner_reset_obs=state.reset_obs
		# TODO: state.info include the outter-wrapper specific key/values, more than state.reset_info which
		# only include the inner-most MjxEnv's info.
		inner_reset_info=state.reset_info

		# kenneth:  EpisodeWrapper and AutoResetWrapper use 'steps',while MjxEnv use 'step'.
		# set 'steps' to zero for EpisodeWrapper. not for MjxEnv.
		# kenneth: we clear 'steps' here instead in EpisodeWrapper.
		if 'steps' in state.info:
			# reset steps to 0 if done.
			steps = state.info['steps']
			steps = jp.where(state.done, jp.zeros_like(steps), steps)
			state.info.update(steps=steps)

		# kenneth: AutoResetWrapper is handler of `done`, so be responsible to clear `done`.
		state = state.replace(done=jp.zeros_like(state.done))

		state = self.env.step(state, action)

		def _where_done(x, y):
			done = state.done
			# kenneth: mjx.Data include _impl:DataJax, which is not vectorized physics data,
			# _impl is jax-specific data, so we should use the _impl from state.data.
			if done.shape and done.shape[0] != x.shape[0]:
				# y is leaf node of state.data/state.obs
				return y
			if done.shape:
				done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
			return jp.where(done, x, y)

		# reset_data , reset_obs is same shape as state.data, state.obs, so we can use jp.where
		# to select directly.
		data = jax.tree.map(_where_done, inner_reset_data, state.data)
		obs = jax.tree.map(_where_done, inner_reset_obs, state.obs)

		def _where_done_select_info(path: Tuple[pytree.DictKey], outer_node: jax.Array) -> jax.Array:
			done=state.done
			# key is "SoftResetWrapper_done_count", "rng", "steps", "last_contact"...
			key:str = path[0].key

			# note: jit sensitive.
			if key not in inner_reset_info:
				# for key belong to outer wrapper, like "SoftResetWrapper_done_count", "episode_metrics",
				# we just return the leaf node of outer info.
				return outer_node

			reset_node = inner_reset_info[key]
			assert reset_node.shape == outer_node.shape

			if done.shape:
				done = jp.reshape(done, [outer_node.shape[0]] + [1] * (len(outer_node.shape) - 1))

			return jp.where(done, reset_node, outer_node)

		# outer_info contains more key/value pairs than inner_reset_info, so we need to select according to info dict key.
		outer_info = state.info
		outer_info = jax.tree.map_with_path(_where_done_select_info, outer_info)

		done_count_key = f'{self._info_key}_done_count'
		# outer_info[done_count_key] = state.info[done_count_key]

		# kenneth: keep 'steps' which used by EpisodeWrapper.
		# TODO: maybe only keep state.info['steps'] for not-done env.
		# if 'steps' in outer_info:
		# 	outer_info['steps'] = state.info['steps']
		# preserve_info_key = f'{self._info_key}_preserve_info'
		# if preserve_info_key in next_info:
		# 	next_info[preserve_info_key] = state.info[preserve_info_key]

		outer_info[done_count_key] += state.done.astype(int)
		# outer_info[f'{self._info_key}_rng'] = reset_rng

		return state.replace(data=data, obs=obs, info=outer_info)
