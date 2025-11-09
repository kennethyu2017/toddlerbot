from typing import Any, Callable, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from functools import partial
from ml_collections import config_dict
import mujoco.mjx as mjx
import jax

from kbot.base_env.base_env_mjx import MjxEnv
from kbot.locomotion.training_params.brax_ppo_params import brax_ppo_config
import kbot.locomotion.kbot_both_leg.training_helper as kbot_both_leg
from kbot.locomotion.kbot_both_leg.randomize import domain_randomize


@dataclass(kw_only=True)
class EnvRegistry:
	env_name: str
	ppo_param_fn: Callable[[], config_dict.ConfigDict]
	train_env_fn: Callable[[], MjxEnv]
	eval_env_fn: Callable[[], MjxEnv]
	rollout_fn: Callable
	render_fn: Callable
	randomization_fn: Callable[[mjx.Model, jax.Array],Tuple[mjx.Model, Any] ]

	def register_to(self, reg_table:Dict[str,"EnvRegistry"]) -> None:
		if self.env_name in reg_table:
			raise KeyError(f"Environment name {self.env_name} already registered")

		reg_table.update({self.env_name: self})


def get_env_registry(env_name: str) -> EnvRegistry:
	registries: Dict[str, EnvRegistry] = {}

	# no need to use global value.
	EnvRegistry(env_name='kbot_both_leg_flat_terrain',
				ppo_param_fn=partial(brax_ppo_config, env_name='kbot_both_leg_flat_terrain'),
				train_env_fn=partial(kbot_both_leg.train_env, task_name='flat_terrain'),
				eval_env_fn=partial(kbot_both_leg.eval_env, task_name='flat_terrain'),
				rollout_fn=kbot_both_leg.rollout,
				render_fn=kbot_both_leg.render_to_video,
				randomization_fn=domain_randomize,
				).register_to(registries)


	if env_name not in registries:
		raise ValueError(f"Unknown env registry of name: {env_name}")

	return registries[env_name]
