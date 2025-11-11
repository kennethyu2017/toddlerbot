
"""Utilities for randomization. Follow mujoco playground g1."""
from typing import Tuple, Any
import jax
from mujoco import mjx
from kbot.base_env.base_env_mjx import MjxEnv
from kbot.locomotion.kbot_both_leg.joystick_env import Joystick

# TO update FLOOR_GEOM_ID = 0
# TO update TORSO_BODY_ID = 16  'torso_link'

def domain_randomize(model: mjx.Model, rng: jax.Array, env:MjxEnv)->Tuple[mjx.Model, Any] :
    if not isinstance(env, Joystick):
        raise NotImplementedError(f'Domain randomization not implemented for {type(env)}, '
                                  f'we only support Joystick env till now.')


    @jax.vmap
    def rand_dynamics(rng):
        # Floor / foot friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(29,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(29,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # # Add mass to torso: +U(-1.0, 1.0).
        # rng, key = jax.random.split(rng)
        # dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        # body_mass = body_mass.at[TORSO_BODY_ID].set(
        #     body_mass[TORSO_BODY_ID] + dmass
        # )

        # kenneth: Add mass to pelvis: +U(-1.0, 1.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        # TODO: we use virtual_floating_base as integrated pelvis temply.
        pelvis_body_id = env._virtual_floating_base_body_id
        body_mass = body_mass.at[pelvis_body_id].set(
            body_mass[pelvis_body_id] + dmass
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05)
        )

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
        )

    (
      pair_friction,
      frictionloss,
      armature,
      body_mass,
      qpos0,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
    })

    model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
    })

    return model, in_axes
