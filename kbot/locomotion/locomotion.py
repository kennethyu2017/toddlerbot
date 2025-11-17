from typing import Callable, List, Dict, Any, Tuple
import time
from datetime import datetime
import functools
import os
import pickle
from functools import partial
from absl import app
from brax.io import model
from brax.training.acme import running_statistics
from ml_collections import config_dict
from etils import epath
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
import numpy as np
import atexit

from tensorboardX import SummaryWriter

# from orbax import checkpoint as ocp
# from vis_utils import display_swing_peak, display_lin_and_angle_vel
from kbot.base_env.base_env_mjx import MjxEnv,State
from kbot.locomotion.env_registry import get_env_registry

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# Enable jax persistent compilation cache.
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", 128)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 4)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

def _default_training_config() -> config_dict.ConfigDict:
    return config_dict.create(
        req_train=True,

        req_eval=True,
        render_rollout=True,

        save_rollout_data=False,
        display_swing_peak=False,

        display_vel=False,

        env_name='TryG1JoystickFlatTerrain',
        # env_name='kbot_both_leg_flat_terrain',
        # env_name='G1JoystickFlatTerrain',

        # TODO: temply debug.
        # env_name='toy_kbot_both_leg_flat_terrain',

        main_seed=423,
        train_seed = 982,
        model_dir = 'models',

        ckpt_root_dir = 'checkpoints',
        # ckpt_root_dir=None,

        video_dir = 'videos',
        rollout_data_dir = 'rollout_data',
        summary_root_dir = 'runs_summary',

        # restore_ckpt = None, # None means not restore.
        restore_ckpt='locomotion_TryG1JoystickFlatTerrain_date_2025_11_17_18_40_19',

        # for easy try.
        ppo_params_override = config_dict.create(
            num_timesteps=200_000_000 // 5,
            num_evals = 20 // 2,
            # for validation(eval) during training epoch.
            num_eval_envs = 32,
            # for training
            num_envs= 8192,

            # network_factory_kwargs=config_dict.create(
            #     policy_hidden_layer_sizes=(512//2, 256//2, 128//2),
            #     value_hidden_layer_sizes=(512//2, 256//2, 128//2),
            #     # must be same as in Observation from MjxEnv.
            #     policy_obs_key="state",
            #     value_obs_key="privileged_state",
            # )
        ),
    )


def _latest_ckpt_path(ckpt_dir: epath.Path | str)->epath.Path:
    FINETUNE_PATH = epath.Path(ckpt_dir)
    latest_ckpts = FINETUNE_PATH.glob("*")
    latest_ckpts = [_ckpt for _ckpt in latest_ckpts if _ckpt.is_dir()]
    # sorte by name which is training step number.
    # latest_ckpts.sort(key=lambda x: int(x.name))
    # return latest_ckpts[-1]
    return max(latest_ckpts, key=lambda _x: _x.name)


def _latest_model_file_path(env_name:str, model_dir: epath.Path)->epath.Path:
    assert model_dir.exists()
    pattern = '{}_{}_date_*.pkl'.format('locomotion', env_name)
    model_file_list = sorted(model_dir.glob(pattern),
                             key=lambda _p:_p.stat().mtime,
                             reverse=True)
    if len(model_file_list) == 0:
        raise ValueError('No model file found')

    return model_file_list[0]


def _gen_model_file_path(env_name:str, model_dir: epath.Path)->epath.Path:
    if not model_dir.exists():
        model_dir.mkdir(parents=False,exist_ok=False)

    model_file = model_dir / '{}_{}_date_{}.pkl'.format('locomotion',
                                                        env_name,
                                                        time.strftime("%Y_%m_%d_%H_%M_%S"))
    return model_file.resolve()


# for tensorboardX.
def _gen_summary_dir(env_name:str, summary_root_dir: epath.Path)->epath.Path:
    summary_dir = (summary_root_dir
                   / '{}_{}'.format('locomotion',env_name)
                   / 'date_{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S")))

    summary_dir.mkdir(parents=not summary_dir.parent.exists(), exist_ok=False)
    return summary_dir.resolve()


def _gen_ckpt_dir(env_name:str, ckpt_root_dir: epath.Path)->epath.Path:
    ckpt_dir = ckpt_root_dir / '{}_{}_date_{}'.format('locomotion',
                                                      env_name,
                                                      time.strftime("%Y_%m_%d_%H_%M_%S"))
    ckpt_dir.mkdir(parents=not ckpt_dir.parent.exists(), exist_ok=False)
    return ckpt_dir.resolve()

def _gen_video_file_path(env_name:str, video_dir: epath.Path)->epath.Path:
    if not video_dir.exists():
        video_dir.mkdir(parents=False, exist_ok=False)

    video_file = video_dir / '{}_{}_date_{}.mp4'.format('locomotion',
                                                        env_name,
                                                        time.strftime("%Y_%m_%d_%H_%M_%S"))
    return video_file.resolve()

def _gen_ro_data_file_path(env_name:str, ro_data_dir: epath.Path)->epath.Path:
    if not ro_data_dir.exists():
        ro_data_dir.mkdir(parents=False, exist_ok=False)

    ro_data_file = ro_data_dir / '{}_{}_date_{}.pkl'.format('locomotion',
                                                                 env_name,
                                                                 time.strftime("%Y_%m_%d_%H_%M_%S"))
    return ro_data_file.resolve()


def evaluate(*,
             env_name:str,
             rng:jax.Array,
             policy_fn:Callable,
             rollout_fn:Callable,
             eval_env_fn:Callable,
             render_fn:Callable,
             save_rollout_data: bool,
             render_rollout: bool,
             ro_data_dir: str,
             video_dir: str,
             )->None:
    eval_env = eval_env_fn()
    print(f"eval env: {env_name}")  #, env_cfg: {eval_env._config}")


    # TODO: single instance of eavl_env, no need to use JIT and GPU. use mujoco CPU will be faster.
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(policy_fn)

    ro_data=rollout_fn(
        env=eval_env,
        # TODO: for evaluate, use mujoco on CPU can be faster than mjx which is suitable for multiple-env-instances.
        jit_reset=jit_reset,
        jit_step=jit_step,
        jit_infer_fn=jit_inference_fn,
        rng=rng,
        record_step_state= render_rollout,
    )
    # TODO: use h5 to save ro data:
    if save_rollout_data:
        ro_data_file = _gen_ro_data_file_path(
            env_name,
            ro_data_dir=epath.Path(ro_data_dir)
        )
        print(f"save rollout data to {ro_data_file}")
        with open(ro_data_file, 'wb') as _f:
            # _f.write(pickle.dumps(ro_data))
            pickle.dump(ro_data, _f)

    if render_rollout:
        video_file = _gen_video_file_path(
            env_name,
            video_dir=epath.Path(video_dir)
        )

        print(f"video_file: {video_file}")
        # no need to change env_cfg. just for render.
        dummy_env: MjxEnv = eval_env_fn()
        # render_fn = render_fn
        render_fn(env=dummy_env, ro_data=ro_data, video_file_path=video_file)


# save ckpt
# def policy_params_fn(current_step, make_policy, params, ckpt_dir):
#     # save checkpoints
#     orbax_checkpointer = orb_ckp.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target(params)
#     ckpt_file = ckpt_dir / f'{current_step}'
#     orbax_checkpointer.save(ckpt_file.resolve(), params, force=True, save_args=save_args)

#  progress_fn not called inside jit-boundary.
# TODO : we can add more 'training/xxx', 'eval/xxx' for interested values.
def _train_progress_fn(num_steps: int,
                       metrics: Dict[str, Any],
                       writer: SummaryWriter,
                       times: List[datetime],
                       ):

    """
    :param writer:
    :param times:
    :param num_steps:
    :param metrics:  keys: 'training/xxx', 'eval/xxx'.

    """
    times.append(datetime.now())

    # TODO: we can add rendered video.
    for k,v in metrics.items():
        writer.add_scalar(
            tag=k,
            scalar_value=v,
            global_step=num_steps,
            display_name = k
        )


def train_policy(*,
                 env_name:str,
                 ppo_params:config_dict.ConfigDict,
                 network_factory:Callable,
                 train_env_fn:Callable,
                 # randomization_fn: Callable[[mjx.Model, jax.Array, MjxEnv], Tuple[mjx.Model, Any]],
                 get_randomization_fn: Callable,
                 get_wrap_env_fn: Callable,
                 ckpt_root_dir:str,
                 restore_ckpt:str = None,
                 model_dir: str,
                 summary_writer:SummaryWriter,
                 train_seed:int,
                 )->None:
    # print(f'{ppo_params=:}')

    # TODO: load env_cfg from config.yaml if required.
    train_env = train_env_fn()
    print(f"train env: {env_name} \n" )
          # f"env_cfg --->\n{train_env._config}")

    ckpt_dir = None
    if ckpt_root_dir is not None:
        # we always make new ckpt dir, even restore ckpt.
        ckpt_dir = _gen_ckpt_dir(
            env_name,
            epath.Path(ckpt_root_dir)
        )
        print(f"ckpt_dir: {ckpt_dir}")

        with open(ckpt_dir / "env_config.yaml", "wt") as _f:
            # yaml.safe_dump(train_env._config.to_dict(), stream=_f, indent=4)
            train_env._config.to_yaml(stream=_f, indent=4)
            print(f'save train env config to {_f.name}')

    else:
        print("=== ckpt_root_dir is None, no ckpt will be saved.===")

    latest_ckpt = None
    if restore_ckpt is not None:
        latest_ckpt = _latest_ckpt_path( epath.Path(ckpt_root_dir) / restore_ckpt ).resolve()
        print(f"=== Restore ckpt from path: {latest_ckpt} ===")

    times = [datetime.now()]
    # NOTE: progress_fn not called inside jit-boundary.
    progress_fn = partial(_train_progress_fn, writer=summary_writer, times=times)

    randomization_fn = partial(get_randomization_fn(), env=train_env)
    wrap_env_fn = get_wrap_env_fn()

    train_fn = functools.partial(
        ppo.train,
        # **dict(ppo_params),
        # use to_dict to resolve recursively with valid references. no use **ppo_params directly?
        **ppo_params.to_dict(),
        environment=train_env,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        episode_length= train_env._config.model.episode_length if 'model' in train_env._config    # kbot env cfg.
                        else train_env._config.episode_length ,  # mujoco playground env cfg.
        # progress_fn not called inside jit-boundary.
        progress_fn=progress_fn,

        # ppo.train use ocp.PyTreeCheckpointer() inside.
        # if save_checkpoint_path is None, no ckpt will be saved.
        save_checkpoint_path=ckpt_dir,

        restore_checkpoint_path=latest_ckpt,  # restore from the checkpoint!
        seed=train_seed,
        run_evals=True,  # will call progress_fn to plot.

        # env for validation. use same as train_env.
        eval_env=train_env,
        # num of eval_envs to use for validation(eval) during training epoch.
        # num_eval_envs=num_valid_envs,
        # eval_env=registry.load(env_name, config=env_cfg),
        # eval_env=valid_env,
        # wrapping domain randomization, vmap, auto-reset, etc.
        # wrap_env_fn=wrapper.wrap_for_brax_training,
        wrap_env_fn=wrap_env_fn,
    )

    make_inference_fn, params, metrics = train_fn()

    if len(times) > 1:
        print(f"time to jit: {times[1] - times[0]}\n"
              f"time to train: {times[-1] - times[1]}")

    # SummaryWriter allow double close.
    summary_writer.close()
    model_file = _gen_model_file_path(env_name,
                                      epath.Path(model_dir))
    model.save_params(model_file.as_posix(), params)
    print(f"=== Save trained model to : {model_file} ===")
    time.sleep(1.0)



def build_eval_policy_fn(*,
                         env_name:str,
                         ppo_params:config_dict.ConfigDict,
                         network_factory:Callable,
                         eval_env_fn:Callable,
                         rng:jax.Array,
                         model_dir:str)->Callable:
    ppo_params.num_timesteps = 0
    ppo_params.num_envs = 1

    # print(f'{ppo_params=:}')

    dummy_env:MjxEnv = eval_env_fn()

    # use unwrapped single env to evaluate.
    env_state:State = dummy_env.reset(rng)

    # Discard the batch axes over devices and envs.
    # obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    # env_state.obs is a dict, as a PyTree container.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, env_state.obs)
    # should be {'state':(xx,), 'privileged_state':(xxx, ) }.
    print(f'{obs_shape=:}')

    # NOTE: same normalize fn as ppo_train.
    # normalize = lambda x, y: x
    normalize_obs_fn = None  # default as identity fn, see make_ppo_networks()
    if ppo_params.normalize_observations:
        print(f'normalize observations is: {ppo_params.normalize_observations} ---> will use running_statistics.normalize for obs.')
        normalize_obs_fn = running_statistics.normalize

    ppo_network = network_factory(
        observation_size=obs_shape,
        action_size=dummy_env.action_size,
        preprocess_observations_fn=normalize_obs_fn,
    )
    make_policy_fn = ppo_networks.make_inference_fn(ppo_network)
    latest_model = _latest_model_file_path(
        env_name,
        epath.Path(model_dir)
    )
    print(f"load latest_model from file: {latest_model}")
    params = model.load_params(latest_model.as_posix())
    policy_fn = make_policy_fn(params, deterministic=True)
    return policy_fn

def gen_summary_writer(env_name:str,
                       summary_root_dir:str)->SummaryWriter:
    summary_dir = _gen_summary_dir(
        env_name=env_name,
        summary_root_dir= epath.Path(summary_root_dir)
    )
    print(f"=== tensorboardX summary write to dir: {summary_dir} ===")

    #TODO: maybe write training config into summary.
    # with open(ckpt_dir / "training_config.yaml", "wt") as _f:
    #     yaml.safe_dump(train_env._config.to_dict(), stream=_f, indent=4)
        # train.to_yaml(stream=_f, indent=4)
        # print(f'save train env config to {_f.name}')

    # Initialize the SummaryWriter
    # will save both to local tensorboardX logdir and comet remote storage.
    # so we can make use of local tensorboard webserver also.
    writer = SummaryWriter(
        logdir=summary_dir.as_posix(),
        comet_config={"disabled": True}
    )
    # writer.add_text("training_config", _default_training_config().to_json_best_effort())
    # can handle double close in writer.close().
    atexit.register(lambda: writer.close())
    return writer



def _main(argv):
    training_cfg =  _default_training_config()
    print(f"training_cfg: {training_cfg}")
    # use key() instead PRNGKey().
    rng = jax.random.key(training_cfg.main_seed)

    env_registry = get_env_registry(training_cfg.env_name)

    ppo_params = env_registry.ppo_param_fn()

    # for easy debug & try.
    if 'ppo_params_override' in training_cfg:
        ppo_params.update(training_cfg.ppo_params_override)

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory_kwargs" in ppo_params:
        # del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory_kwargs
        )
        # ppo.train() does not need network_factory_kwargs.
        del ppo_params.network_factory_kwargs
    else:
        raise ValueError("network_factory_kwargs is not defined, do not use the default args of ppo_networks.make_ppo_networks")

    summary_writer = gen_summary_writer(training_cfg.env_name, training_cfg.summary_root_dir)
    summary_writer.add_text("training_config", training_cfg.to_json_best_effort(indent=2))

    if training_cfg.req_train:
        print('=== start train ===')
        train_policy(env_name=training_cfg.env_name,
                     ppo_params=ppo_params,
                     network_factory=network_factory,
                     # ckpt_dir=restore_ckpt_dir if RESTORE_CKPT else None,
                     train_env_fn=env_registry.train_env_fn,
                     train_seed=training_cfg.train_seed,
                     get_randomization_fn=env_registry.get_domain_randomizer,
                     get_wrap_env_fn=env_registry.get_env_wrapper,
                     restore_ckpt=training_cfg.restore_ckpt,
                     ckpt_root_dir=training_cfg.ckpt_root_dir,
                     model_dir=training_cfg.model_dir,
                     summary_writer=summary_writer,
                     )

    # else:
    #     print('=== task: eval only ===')
    #     policy_fn = build_eval_policy_fn(ppo_training_params=ppo_training_params,
    #                                      network_factory=network_factory,
    #                                      env_name=ENV_NAME
    #                                      )

    if training_cfg.req_eval:
        print(f'=== start evaluate, rollout and render video ===')
        # always load params from saved model file, even eval just after training.
        rng, key_policy, key_eval = jax.random.split(rng, 3)

        policy_fn = build_eval_policy_fn(env_name=training_cfg.env_name,
                                         ppo_params=ppo_params,
                                         network_factory=network_factory,
                                         rng=key_policy,
                                         eval_env_fn=env_registry.eval_env_fn,
                                         model_dir=training_cfg.model_dir
                                         )

        evaluate(
            policy_fn=policy_fn,
            env_name=training_cfg.env_name,
            rng=key_eval,
            rollout_fn=env_registry.rollout_fn,
            eval_env_fn=env_registry.eval_env_fn,
            save_rollout_data=training_cfg.save_rollout_data,
            render_rollout=training_cfg.render_rollout,
            render_fn=env_registry.render_fn,
            ro_data_dir=training_cfg.rollout_data_dir,
            video_dir=training_cfg.video_dir,
        )


        # dummy_env_cfg = registry.get_default_config(ENV_NAME)
        # if display_swing_peak:
        #     display_swing_peak(env_cfg=dummy_env_cfg,swing_peak=ro_data['swing_peak'])
        #
        # if display_vel:
        #     display_lin_and_angle_vel(env_cfg=dummy_env_cfg,
        #                               lin_vel=ro_data['lin_vel'],
        #                               ang_vel=ro_data['ang_vel'],
        #                               command=ro_data['command'])


if __name__ == '__main__':
    # import matplotlib
    # print(f"Current matplotlib backend: {matplotlib.get_backend()}")
    # time.sleep(1.)
    # exit(0)

    app.run(_main)


