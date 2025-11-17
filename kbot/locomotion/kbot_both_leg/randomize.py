
"""Utilities for randomization. Follow mujoco playground g1."""
from typing import Tuple, Any
import jax
from mujoco import mjx
from kbot.base_env.base_env_mjx import MjxEnv
from kbot.locomotion.kbot_both_leg.joystick.joystick_env import Joystick

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
            # key, shape=(29,), minval=0.5, maxval=2.0
            key, shape=(model.nu,), minval=0.5, maxval=2.0
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(model.nu,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # # Add mass to torso: +U(-1.0, 1.0). g1 torso_link 7kg.
        # rng, key = jax.random.split(rng)
        # dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        # body_mass = body_mass.at[TORSO_BODY_ID].set(
        #     body_mass[TORSO_BODY_ID] + dmass
        # )

        # kenneth: Add mass to l and r pelvis: +U(-.3, .3):  g1 torso_link 7kg, k-bot left_pelvis or right_pelvis is 2kg.
        rng, key = jax.random.split(rng)
        # dmass = jax.random.uniform(key, minval=-.3, maxval=.3)
        for _id in  env._body_id.pelvis_body_id:
            # TODO: let left right different dmass.
            dmass = jax.random.uniform(key, minval=-.3, maxval=.3)
            body_mass = body_mass.at[_id].set(
                body_mass[_id] + dmass
            )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(key, shape=(model.nu,), minval=-0.05, maxval=0.05)
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


if __name__ == '__main__':
    import jax.numpy as jp

    num_envs = 1024

    from kbot.locomotion.kbot_both_leg.joystick_env import Joystick
    env = Joystick('flat_terrain')
    rng = jax.random.key(0)
    rng, *random_keys = jax.random.split(rng, 1024 + 1)
    random_keys = jax.numpy.array(random_keys)
    # print(env._mjx_model.nu,env._mjx_model.njnt)

    model_v, model_in_axis = domain_randomize(env._mjx_model, random_keys, env)
    print(f'{model_v.qpos0.shape=:} {model_v.dof_frictionloss.shape=:}' )
    print(f'{model_v.jnt_actfrcrange.shape=:}')
    print(f'{model_in_axis.qpos0=:} {model_in_axis.dof_frictionloss=:}')
    print(f'{model_in_axis.jnt_actfrcrange=:}')


    def _check_nan(x: jax.Array):
        has_nan = jp.any(jp.isnan(x))
        assert not has_nan
        return has_nan

    def _check_shape_and_dtype(x: jax.Array):
        return jax.eval_shape(lambda: x)


    def _check_shape(x: jax.Array):
        return x.shape

    # print(jax.tree.map(_check_shape, model_v))
    # print(f'{model_in_axis=:}')

