"""Wrappers for MuJoCo Playground environments."""
from typing import Any, Tuple, Optional,Callable
import jax
from jax import numpy as jp
from jax._src.lib import pytree
import mujoco.mjx as mjx
from mujoco_playground._src import wrapper
from brax.envs.wrappers import training as brax_training
from base_env_mjx import State, MjxEnv

def wrap_for_locomotion_training(
	env: MjxEnv,
	episode_length: int = 1000,
	action_repeat: int = 1,
	randomization_fn: Optional[
		Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
	] = None,
	full_reset: bool = False,
) -> wrapper.Wrapper:
	"""Common wrapper pattern for all brax training agents.

	Args:
	env: environment to be wrapped
	episode_length: length of episode
	action_repeat: how many repeated actions to take per step
	randomization_fn: randomization function that produces a vectorized model
	  and in_axes to vmap over
	full_reset: whether to call `env.reset` during `env.step` on done rather
	  than resetting to a cached first state. Setting full_reset=True may
	  increase wallclock time because it forces full resets to random states.

	Returns:
	An environment that is wrapped with Episode and AutoReset wrappers.  If the
	environment did not already have batch dimensions, it is additional Vmap
	wrapped.
	"""
	del full_reset
	if randomization_fn is None:
		# env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
		raise NotImplementedError('Randomization function must be provided.')

	#kenneth: vectorize multiple envs.
	env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)

	# kenneth: increment info['steps'], and set state.done if 'steps' > episode_length.
	env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)

	# kenneth: AutoResetWrapper is the handler of `done`, also be responsible to clear `done` after process.
	# env = BraxAutoResetWrapper(env, full_reset=full_reset)
	env = SoftResetWrapper(env)
	return env


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

	def __init__(self, env: Any):
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
			# clear info['steps'] if last step() got done.
			steps = jp.where(state.done, jp.zeros_like(steps), steps)
			state.info.update(steps=steps)

		# kenneth: AutoResetWrapper is handler of `done`, so be responsible to clear `done`.
		# kenneth: NOTE, we clear state.done here, just before 1st step() after reset, not in the
		# step() which causing `done`, because after exit from SoftResetWrapper.step(), outer_wrapper,
		# e.g. EvalWrapper will make use of state.done.
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

		# kenneth: NOTE, we can not clear state.done immediately after soft_reset,
		# cause outer_wrapper, e.g. EvalWrapper will make use of state.done.
		return state.replace(data=data, obs=obs, info=outer_info)


if __name__ == '__main__':
	import os
	# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
	xla_flags = os.environ.get('XLA_FLAGS', '')
	xla_flags += ' --xla_gpu_triton_gemm_any=True'
	os.environ['XLA_FLAGS'] = xla_flags

	# Enable jax persistent compilation cache.
	# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
	# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
	# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
	jax.config.update("jax_compilation_cache_dir", "./jax_cache")
	jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
	jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
	# More legible printing from numpy.
	import numpy as np
	np.set_printoptions(precision=3, suppress=True, linewidth=100)

	from kbot.locomotion.kbot_both_leg.env_cfg import default_config
	from kbot.locomotion.kbot_both_leg.joystick_env import Joystick
	from kbot.locomotion.kbot_both_leg.randomize import domain_randomize
	from functools import partial

	task_name = 'flat_terrain'
	num_envs = 3

	toy_cfg = default_config(task_name)
	# in-place
	toy_cfg.update_from_flattened_dict({
		'command.resample_length':1,
		'model.episode_length':10,
	})
	print(f'{toy_cfg=:}')
	mjx_env=Joystick('flat_terrain', config=toy_cfg)

	def _print_state_of_env_0(state_env_0:State):
		print(f' state of env 0 ---> ')
		print(f' qacc: {state_env_0.data.qacc}')
		print(f' qvel: {state_env_0.data.qvel}')
		print(f' sensordata: {state_env_0.data.sensordata}')

		print(f'info ----> ', state_env_0.info)
		print(f'reset info ----> ', state_env_0.reset_info)
		print('obs["state"]:', state_env_0.obs['state'])
		print('done:', state_env_0.done)
		print('rwd ', state_env_0.reward)


	rng = jax.random.key(0)
	# rng, key = jax.random.split(rng)
	# _print_state_of_env_0(mjx_env.reset(key))
	# exit(0)

	rng = jax.random.split(rng, num_envs)
	rng_key = jax.vmap(partial(jax.random.split, num=3) )(rng)
	rng, key_domain_random, key_reset = rng_key.T[0], rng_key.T[1], rng_key.T[3]

	wrapped_env = wrap_for_locomotion_training(env=mjx_env,
											   episode_length=toy_cfg.model.episode_length,
											   action_repeat=1,
											   randomization_fn=partial(domain_randomize, rng=key_domain_random, env=mjx_env),
											   )
	# print(f'{wrapped_env._info_key=:}')
	# print(f'{wrapped_env._mjx_model_v.qpos0.shape=:}')

	reset_state = wrapped_env.reset(key_reset)

	# # print(jax.tree_util.tree_structure(reset_state))
	# print(f'{reset_state.data.qpos.shape=:}')
	# print(f'{reset_state.obs["state"].shape=:}')
	# print(f'{reset_state.info.keys()=:}')
	# print(f'{reset_state.metrics.keys()=:}')

	print('=== reset state ===')
	_print_state_of_env_0(jax.tree.map(lambda x: x[0], reset_state))


	# for _cnt in range(20):
	# 	dummy_action=jp.zeros_like(reset_state.data.ctrl)
	# 	step_state = wrapped_env.step(reset_state, dummy_action)
	# 	if _cnt % 10 == 0:
	# 		print(f'=== step {_cnt} state ===')
	# 		_print_state_of_env_0(step_state)










