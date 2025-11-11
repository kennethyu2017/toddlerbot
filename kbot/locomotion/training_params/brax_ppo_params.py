"""RL config for Locomotion envs. Follow mujoco playground."""

from typing import Optional
from ml_collections import config_dict

def brax_ppo_config(
    env_name: str,
    impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  # env_config = locomotion.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=100_000_000,

      # total validation(eval) during entire training.so we got running progress_fn 10 times.
      num_evals=10,
      # for validation(eval) during training epoch. default 128 in ppo.train().
      num_eval_envs=32,

      reward_scaling=1.0,
      # episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      # argument for ppo_networks.make_ppo_networks()
      network_factory_kwargs=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
      # full reset per training validation(eval).
      num_resets_per_eval=10,
  )

  if env_name in ("kbot_both_leg_flat_terrain", "kbot_both_leg_rough_terrain"):
    rl_config.num_timesteps = 200_000_000

    # total validation(eval) during entire training.so we got running progress_fn 20 times.
    rl_config.num_evals = 40 #20
    # for validation(eval) during training epoch.
    rl_config.num_eval_envs = 64

    rl_config.clipping_epsilon = 0.2

    # full reset per training validation(eval).
    rl_config.num_resets_per_eval = 1

    rl_config.entropy_cost = 0.005
    # argument for ppo_networks.make_ppo_networks()
    rl_config.network_factory_kwargs=config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        # must be same as in Observation from MjxEnv.
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config



# for debug only.
def toy_brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
    rl_config = brax_ppo_config(env_name)

    if env_name in ("kbot_both_leg_flat_terrain", "kbot_both_leg_rough_terrain"):
        rl_config.update(
            num_timesteps=1024,
            num_envs=16,
            num_evals=10,
            num_eval_envs=8,
            batch_size=16,
            num_minibatches=8,
            num_resets_per_eval=1,
            network_factory_kwargs = config_dict.create(
                policy_hidden_layer_sizes=(32, 32, 16),
                value_hidden_layer_sizes=(32, 32, 16),
                # must be same as in Observation from MjxEnv.
                policy_obs_key="state",
                value_obs_key="privileged_state")
        )

    return rl_config


if __name__ == "__main__":
    ppo_params = brax_ppo_config('kbot_both_leg_flat_terrain')
    # print(f'{ppo_params=:}')
    print(f'{ppo_params.to_dict()=:}')

    toy_params = toy_brax_ppo_config('kbot_both_leg_rough_terrain')
    print(f'{toy_params.to_dict()=:}')

    # network_factory_args_dict = {}
    # network_factory_args_dict.update(**ppo_params.network_factory_kwargs)
    # print(f'{network_factory_args_dict=:}')
    # print(f'{ppo_params.keys()=:}')
    # print('entropy_cost' in ppo_params )

