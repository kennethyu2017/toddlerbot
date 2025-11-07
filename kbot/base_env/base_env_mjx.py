# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core classes for MuJoCo Playground."""

import abc
import unittest
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from etils import epath
from flax import struct
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
import tqdm
from functools import partial

Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

def update_assets(
    assets: Dict[str, Any],
    path: Union[str, epath.Path],
    glob: str = "*",
    recursive: bool = False,
):
  for f in epath.Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)


def make_mjx_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    #nconmax,njmax are deprecated only use for mujoco prior to 2.3.0.
    # and only used for `warp` backend.
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  data = mjx.make_data(
      model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
  )
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  if act is not None:
    data = data.replace(act=act)
  if mocap_pos is not None:
    data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
  if mocap_quat is not None:
    data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
  return data


# kenneth: flax.struct.dataclass is a 'frozen' dataclass, so must use .replace() to modify member.
# note: .replace() do shallow copy on other non_replaced data members, so it is efficient to do .replace() multiple times.
# especially used in jit.
@partial(struct.dataclass, kw_only=True)
class State:
  """Environment state for training and inference."""

  data: mjx.Data
  obs: Observation
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array]
  info: Dict[str, Any]

  def tree_replace(
      self, params: Dict[str, Optional[jax.typing.ArrayLike]]
  ) -> "State":
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split("."), v)
    return new

# State = struct.dataclass(__State, kw_only=True)

def _tree_replace(
    base: Any,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Any:
  """Sets attributes in a struct.dataclass with values."""
  if not attr:
    return base

  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    raise NotImplementedError("List attributes are not supported.")

  if len(attr) == 1:
    return base.replace(**{attr[0]: val})

  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )

def get_assets(xml_path: epath.Path) -> Dict[str, bytes]:
  assets = {}
  update_assets(assets, xml_path, "*.xml")
  update_assets(assets, xml_path / "assets", recursive=True)
  return assets


class MjxEnv(abc.ABC):
  """Base class for playground environments."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    self._config = config.lock()
    if config_overrides:
      self._config.update_from_flattened_dict(config_overrides)

    xml_dir = epath.Path(self._config.model.xml_dir)
    xml_file = xml_dir / self._config.model.xml_file
    self._model_assets = get_assets(xml_dir)
    self._mj_model:mujoco.MjModel = mujoco.MjModel.from_xml_string(
      xml=xml_file.read_text(), assets=self._model_assets
    )

    self._mj_model.opt.timestep = self._config.model.sim_dt

    if self._config.model.restricted_joint_range:
      # inheritrange=1
      self._mj_model.jnt_range[1:] = self._config.model.robot.restricted_joint_range
      self._mj_model.actuator_ctrlrange[:] = self._config.model.robot.restricted_joint_range

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.model.impl)
    self._xml_file = xml_file.resolve()

  """
  interface functions.
  """
  @abc.abstractmethod
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

  @staticmethod
  def jax_step(
          model: mjx.Model,
          data: mjx.Data,
          # state: State,
          action: jax.Array,
          n_substeps: int = 1,
  ) -> mjx.Data:
    def single_step(_d: mjx.Data, _)-> Tuple[mjx.Data, Any]:
      _d = _d.replace(ctrl=action)
      _d = mjx.step(model, _d)
      return _d, None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]
    # new_data=jax.lax.scan(single_step, state.data, (), n_substeps)[0]
    # return state.replce(data=new_data)

  """
  accessors.
  """
  @property
  def unwrapped(self) ->"MjxEnv":
    return self

  @property
  def xml_file(self) -> str:
    return self._config.model.xml_file

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def ctrl_dt(self) -> float:
    """Control timestep for the environment."""
    return self._config.model.ctrl_dt

  @property
  def sim_dt(self) -> float:
    """Simulation timestep for the environment."""
    return self._config.model.sim_dt

  @property
  def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.ctrl_dt / self.sim_dt))

  @property
  def observation_size(self) -> ObservationSize:
    abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
    obs = abstract_state.obs
    if isinstance(obs, Mapping):
      return jax.tree_util.tree_map(lambda x: x.shape, obs)
    return obs.shape[-1]

  @property
  def model_assets(self) -> Dict[str, Any]:
    """Dictionary of model assets to use with MjModel.from_xml_path."""
    if hasattr(self, "_model_assets"):
      return self._model_assets
    raise NotImplementedError(
        "_model_assets not defined for this environment"
        "see cartpole.py for an example."
    )

  """
  Sensor readings.
  """
  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.robot.gravity_sensor}_{frame}"
    )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.robot.global_linvel_sensor}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.robot.global_angvel_sensor}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.model.robot.local_linvel_sensor}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.model.robot.accelerometer_sensor}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return get_sensor_data(
      self.mj_model, data, f"{self._config.model.robot.gyro_sensor}_{frame}"
    )

  def render(
      self,
      trajectory: List[State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return render_array(
        self.mj_model,
        trajectory,
        height,
        width,
        camera,
        scene_option=scene_option,
        modify_scene_fns=modify_scene_fns,
    )


def render_array(
    mj_model: mujoco.MjModel,
    trajectory: Union[List[State], State],
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[
        Sequence[Callable[[mujoco.MjvScene], None]]
    ] = None,
    hfield_data: Optional[jax.Array] = None,
):
  """Renders a trajectory as an array of images."""
  renderer = mujoco.Renderer(mj_model, height=height, width=width)
  camera = camera if camera is not None else -1

  if hfield_data is not None:
    mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
    mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

  def get_image(state, modify_scn_fn=None) -> np.ndarray:
    d = mujoco.MjData(mj_model)
    d.qpos, d.qvel = state.data.qpos, state.data.qvel
    d.mocap_pos, d.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
    d.xfrc_applied = state.data.xfrc_applied
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
    if modify_scn_fn is not None:
      modify_scn_fn(renderer.scene)
    return renderer.render()

  if isinstance(trajectory, list):
    out = []
    for i, state in enumerate(tqdm.tqdm(trajectory)):
      if modify_scene_fns is not None:
        modify_scene_fn = modify_scene_fns[i]
      else:
        modify_scene_fn = None
      out.append(get_image(state, modify_scene_fn))
  else:
    out = get_image(trajectory)

  renderer.close()
  return out


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
  """Gets sensor data given sensor name."""
  # sensor_id = model.sensor(sensor_name).id
  # sensor_adr = model.sensor_adr[sensor_id]
  # sensor_dim = model.sensor_dim[sensor_id]
  sensor_view = model.sensor(sensor_name)
  sensor_adr = sensor_view.adr[0]
  sensor_dim = sensor_view.dim[0]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qvel."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
  """Get the dimensionality of the joint in qpos."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> np.ndarray:
  index_list: list[int] = []
  for jnt_name in joint_names:
    jnt = model.joint(jnt_name).id
    jnt_type = model.jnt_type[jnt]
    qadr = model.jnt_qposadr[jnt]
    qdim = qpos_width(jnt_type)
    index_list.extend(range(qadr, qadr + qdim))
  return np.array(index_list)


def get_qvel_ids(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> np.ndarray:
  index_list: list[int] = []
  for jnt_name in joint_names:
    jnt = model.joint(jnt_name).id
    jnt_type = model.jnt_type[jnt]
    vadr = model.jnt_dofadr[jnt]
    vdim = dof_width(jnt_type)
    index_list.extend(range(vadr, vadr + vdim))
  return np.array(index_list)


if __name__ == '__main__':
  from kbot.locomotion.kbot_both_leg.env_cfg import default_config
  class TestEnv(MjxEnv):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def reset(self, rng: jax.Array):
      pass

    def step(self, state: State, action: jax.Array):
      pass

  test_cfg = default_config('flat_terrain')
  test_env = TestEnv(test_cfg)
  print(f'{test_env.n_substeps=:}')
  test_mjx_data=mjx.make_data(test_env.mjx_model)
  print('pelvis accelerometer sensor data:', test_env.get_accelerometer(test_mjx_data,'pelvis'))
  print('pelvis gyro sensor data:', test_env.get_gyro(test_mjx_data, 'pelvis'))