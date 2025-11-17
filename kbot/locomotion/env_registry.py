from typing import Any, Callable, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from functools import partial
from ml_collections import config_dict
import mujoco.mjx as mjx
import jax

import kbot.locomotion.kbot_both_leg.joystick.try_g1_joystick_rwd
from kbot.base_env.base_env_mjx import MjxEnv
from kbot.locomotion.training_params.brax_ppo_params import brax_ppo_config, toy_brax_ppo_config
import kbot.locomotion.kbot_both_leg.training_helper as kbot_both_leg

# import kbot.locomotion.g1.G1_joystick_train_helper as G1_joystick
import kbot.locomotion.kbot_both_leg.try_g1_training_helper as TryG1_joystick


from mujoco_playground.config import locomotion_params

@dataclass(kw_only=True)
class EnvRegistry:
	env_name: str
	ppo_param_fn: Callable[[], config_dict.ConfigDict]
	train_env_fn: Callable[[], MjxEnv]
	eval_env_fn: Callable[[], MjxEnv]
	rollout_fn: Callable
	render_fn: Callable
	# randomization_fn: Callable[[mjx.Model, jax.Array, MjxEnv],Tuple[mjx.Model, Any] ]
	get_domain_randomizer:  Callable
	get_env_wrapper: Callable

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
				get_domain_randomizer=partial(kbot_both_leg.get_domain_randomizer,env_name='kbot_both_leg_flat_terrain'),
				get_env_wrapper=partial(kbot_both_leg.get_env_wrapper,env_name='kbot_both_leg_flat_terrain'),
				).register_to(registries)

	# EnvRegistry(env_name='toy_kbot_both_leg_flat_terrain',
	# 			ppo_param_fn=partial(toy_brax_ppo_config, env_name='kbot_both_leg_flat_terrain'),
	# 			train_env_fn=partial(kbot_both_leg.train_env, task_name='flat_terrain'),
	# 			eval_env_fn=partial(kbot_both_leg.eval_env, task_name='flat_terrain'),
	# 			rollout_fn=kbot_both_leg.rollout,
	# 			render_fn=kbot_both_leg.render_to_video,
	# 			randomization_fn= kbot_both_leg.get_randomizer('toy_kbot_both_leg_flat_terrain'),
	# 			).register_to(registries)

	# === debug compare only.
	# EnvRegistry(env_name='G1JoystickFlatTerrain',
	# 			ppo_param_fn=partial(locomotion_params.brax_ppo_config,'G1JoystickFlatTerrain'),
	# 			train_env_fn=partial(G1_joystick.train_env, env_name='G1JoystickFlatTerrain'),
	# 			eval_env_fn=partial(G1_joystick.eval_env, env_name='G1JoystickFlatTerrain'),
	# 			rollout_fn=G1_joystick.rollout,
	# 			render_fn=G1_joystick.render_to_video,
	# 			get_domain_randomizer=partial(G1_joystick.get_domain_randomizer, 'G1JoystickFlatTerrain'),
	# 			get_env_wrapper=partial(G1_joystick.get_env_wrapper,'G1JoystickFlatTerrain'),
	# 			).register_to(registries)

	EnvRegistry(env_name='TryG1JoystickFlatTerrain',
				ppo_param_fn=partial(locomotion_params.brax_ppo_config, 'G1JoystickFlatTerrain'),
				train_env_fn=partial(TryG1_joystick.train_env, env_name='G1JoystickFlatTerrain'),
				eval_env_fn=partial(TryG1_joystick.eval_env, env_name='G1JoystickFlatTerrain'),
				rollout_fn=TryG1_joystick.rollout,
				render_fn=TryG1_joystick.render_to_video,
				get_domain_randomizer=partial(TryG1_joystick.get_domain_randomizer, 'G1JoystickFlatTerrain'),
				get_env_wrapper=partial(TryG1_joystick.get_env_wrapper, 'G1JoystickFlatTerrain'),
				).register_to(registries)

	if env_name not in registries:
		raise ValueError(f"Unknown env registry of name: {env_name}")

	return registries[env_name]

if __name__ == '__main__':
	env_registry = get_env_registry('kbot_both_leg_flat_terrain')
	print(env_registry)
