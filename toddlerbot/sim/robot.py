import json
from typing import Any, List, Mapping, Tuple, OrderedDict, Sequence, Dict
from collections import OrderedDict as OrdDictCls
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from pathlib import Path

from ._module_logger import logger

@dataclass(init=True)
class _RobotCfgData:
    """
    only define data attributes.
      init_motor_angles (OrderedDict[str, float]): Initial motor angles for active joints.
      init_active_joint_angles (OrderedDict[str, float]): Initial joint angles derived from motor angles.
      motor_name_ordering (List[str]): Order of motors name based on initial configuration.
      motor_id_ordering (List[str]): Order of motors id based on initial configuration.
      active_joint_name_ordering (List[str]): Order of joints based on initial configuration.
      default_motor_angles (OrderedDict[str, float]): Default motor angles for active joints.
      default_active_joint_angles (OrderedDict[str, float]): Default joint angles derived from motor angles.
      motor_to_active_joint_name (OrderedDict[str, List[str]]): Mapping from motor names to joint names.
      active_joint_to_motor_name (OrderedDict[str, List[str]]): Mapping from joint names to motor names.
      passive_joint_names (List[str]): Names of passive joints.
      foot_name (str): Name of the foot, if specified in the configuration.
      has_gripper (bool): Indicates if the robot has a gripper.
      collider_names (List[str]): Names of links with collision enabled.
      nu (int): Number of active motors.
      joint_cfg_groups (OrderedDict[str, str]): Group classification for each joint.
      joint_cfg_limits (OrderedDict[str, List[float]]): Lower and upper limits for each joint.
      """
    # id: int
    name: str = ''
    # nu: int
    # foot_name: strget_joint_attrs
    has_gripper: bool = False
    collider_names: List[str] = field(default_factory=list)

    joint_cfg_groups: OrderedDict[str, List[float]] = field(default_factory=OrdDictCls)
    joint_cfg_limits: OrderedDict[str, List[float]] = field(default_factory=OrdDictCls)

    passive_joint_names: List[str] = field(default_factory=list)
    active_joint_to_motor_name: OrderedDict[str, List[str]] = field(default_factory=OrdDictCls)
    motor_to_active_joint_name: OrderedDict[str, List[str]] = field(default_factory=OrdDictCls)

    # for actuators. not passive.

    # TODO: rename it to motor_joint_name_to_id....

    motor_name_to_id: OrderedDict[str, int] = field(default_factory=OrdDictCls)
    id_to_motor_name: OrderedDict[int, str] = field(default_factory=OrdDictCls)

    default_motor_angles: OrderedDict[str, float] = field(default_factory=OrdDictCls)
    default_active_joint_angles: OrderedDict[str, float] = field(default_factory=OrdDictCls)

    # keep motor_name and motor_id as same order.
    motor_name_ordering: Tuple[str, ...] = field(default_factory=tuple)
    motor_id_ordering: Tuple[int, ...] = field(default_factory=tuple)

    active_joint_name_ordering: Tuple[str, ...] = field(default_factory=tuple)

    init_active_joint_angles: OrderedDict[str, float] = field(default_factory=OrdDictCls)
    init_motor_angles: OrderedDict[str, float] = field(default_factory=OrdDictCls)

    config: Mapping[str, Any] = None
    collision_config: Mapping[str, Any] = None

    # TODO: motor params.integrate into one struct.
    motor_type: OrderedDict[str, str] = field(default_factory=OrderedDict)
    motor_kp_real: OrderedDict[str, float] = field(default_factory=OrderedDict)
    motor_kd_real: OrderedDict[str, float] = field(default_factory=OrderedDict)
    motor_ki_real: OrderedDict[str, float] = field(default_factory=OrderedDict)

    motor_kp_sim: OrderedDict[str, float] = field(default_factory=OrderedDict)
    motor_kd_sim: OrderedDict[str, float] = field(default_factory=OrderedDict)

    motor_tau_max: OrderedDict[str, float] = field(default_factory=OrderedDict)
    motor_q_dot_tau_max: OrderedDict[str, float] = field(default_factory=OrderedDict)
    motor_q_dot_max: OrderedDict[str, float] = field(default_factory=OrderedDict)


class Robot(_RobotCfgData):
    """This class defines some data structures, FK, IK of ToddlerBot."""


    def __init__(self, robot_name: str):
        """Initializes a robot with specified configurations and paths.

        Args:
            robot_name (str): The name of the robot, used to set up directory paths and configurations.
        """
        super().__init__()
        self.name = robot_name
        self.root_path: Path = Path('toddlerbot')/'descriptions'/self.name  #  os.path.join("toddlerbot", "descriptions", self.name)
        self.config_path: Path = self.root_path / 'config.json'
        self.collision_config_path :Path = self.root_path /'config_collision.json'

        self.cache_path = self.root_path / f'{self.name}_cache.pkl'

        self.load_robot_config()

        self.initialize()

    def load_robot_config(self):
        """Load the robot's configuration and collision configuration from JSON files.

        Raises:
            FileNotFoundError: If the main configuration file or the collision configuration file does not exist at the specified paths.
        """
        # if os.path.exists(self.config_path):
        if self.config_path.exists():
            with open(self.config_path, "rt") as f:
                self.config = json.load(f)

        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")

        # if os.path.exists(self.collision_config_path):
        if self.collision_config_path.exists():
            with open(self.collision_config_path, "rt") as f:
                self.collision_config = json.load(f)

        else:
            raise FileNotFoundError(
                f"No collision config file found for robot '{self.name}'."
            )


    def _parse_motor_param_helper(self, name: str, cfg: Mapping[str, float|str])->None:

        # TODO: motor params.integrate into one struct.

        #  'dynamixel' or 'feite'
        self.motor_type[name] = cfg['type']

        self.motor_kp_real[name] = cfg['kp_real']
        self.motor_kd_real[name] = cfg['kd_real']
        self.motor_ki_real[name] = cfg['ki_real']

        self.motor_kp_sim[name] = cfg['kp_sim']
        self.motor_kd_sim[name] = cfg['kd_sim']

        self.motor_tau_max[name] = cfg['tau_max']
        self.motor_q_dot_tau_max[name] = cfg['q_dot_tau_max']
        self.motor_q_dot_max[name] = cfg['q_dot_max']


    def initialize(self) -> None:
        """Initializes the robot's joint and motor configurations based on the provided configuration data.

        This method sets up the initial and default motor angles, establishes mappings between motors and joints,
         identifies passive joints, and configures collision detection. It also determines the presence of
          a gripper and sets joint groups and limits.

        Attributes:
            init_motor_angles (OrderedDict[str, float]): Initial motor angles for active joints.
            init_active_joint_angles (OrderedDict[str, float]): Initial joint angles derived from motor angles.
            motor_name_ordering (List[str]): Order of motors name based on initial configuration.
            motor_id_ordering (List[int]): Order of motors id based on initial configuration.
            active_joint_name_ordering (List[str]): Order of joints based on initial configuration.
            default_motor_angles (OrderedDict[str, float]): Default motor angles for active joints.
            default_active_joint_angles (OrderedDict[str, float]): Default joint angles derived from motor angles.
            motor_to_active_joint_name (OrderedDict[str, List[str]]): Mapping from motor names to joint names.
            active_joint_to_motor_name (OrderedDict[str, List[str]]): Mapping from joint names to motor names.
            passive_joint_names (List[str]): Names of passive joints.
            foot_name (str): Name of the foot, if specified in the configuration.
            has_gripper (bool): Indicates if the robot has a gripper.
            collider_names (List[str]): Names of links with collision enabled.
            nu (int): Number of active motors.
            joint_cfg_groups (OrderedDict[str, str]): Group classification for each joint.
            joint_cfg_limits (OrderedDict[str, List[float]]): Lower and upper limits for each joint.
        """
        # self.init_motor_angles = {}
        # parse motors:
        for _name, _cfg in self.config["joints"].items():
            self.joint_cfg_groups[_name] = _cfg["group"]
            self.joint_cfg_limits[_name] = [
                _cfg["lower_limit"],
                _cfg["upper_limit"],
            ]
            # TODO: to be not confused with the passive_joint.
            #  we can rename  "is_passive": false to 'motor_drive': true.
            if not _cfg["is_passive"]:
                # when a config joint is not passive, use it as `motor`.
                # self.init_motor_angles[joint_name] = 0.0
                # self.default_motor_angles[joint_name] = joint_config["default_pos"]

                # check duplicated:
                if _name in self.motor_name_to_id:
                    raise ValueError(f'joint name duplicated: {_name}')

                motor_id:int = _cfg['id']
                if  motor_id in self.id_to_motor_name:
                    raise ValueError(f'config motor id duplicated: cfg joint name: {_name}, cfg id: {motor_id}')

                self.motor_name_to_id[_name] = motor_id
                self.id_to_motor_name[motor_id] = _name

                # TODO: why not read init_pos from cfg ???
                # self.init_motor_angles[_name] = 0.0
                self.init_motor_angles[_name] = _cfg['init_pos']

                self.default_motor_angles[_name] = _cfg['default_pos']

                self._parse_motor_param_helper(_name,_cfg)

        # NOTE: keep same ordering.
        self.motor_name_ordering = tuple(self.motor_name_to_id)
        self.motor_id_ordering = tuple(self.motor_name_to_id.values())

        # generate joints based on motors.
        # NOTE: the keys in `init_active_joint_angles` maybe more than the `joint` and motors in config.
        # NOTE: here, the `active joint` is derived from motor, not same as in config, not self.passive_joints which
        # only include `linkage` and `rack_and_pinion` transmission.
        # TODO: better naming `active` and `passive`... NOTE: a joint driven by gear, e.g., right_elbow_yaw_driven, even whose `is_passive` is true in config.json,
        # is still one of self.active_joint, and is not `self.passive_joint`.
        self.init_active_joint_angles = self.motor_to_active_joint_angles(self.init_motor_angles)

        self.default_active_joint_angles = self.motor_to_active_joint_angles(self.default_motor_angles)

        #NOTE: len(self.motor_name_ordering) != len(self.active_joint_name_ordering)
        # immutable.
        # self.motor_name_ordering = tuple(self.init_motor_angles)  #  list(self.init_motor_angles) #.keys())
        # self.active_joint_name_ordering = tuple(self.init_active_joint_angles)  #  list(self.init_active_joint_angles) #.keys())

        logger.info(f' robot motor name ordering: {self.motor_name_ordering}  len:{len(self.motor_name_ordering)}')
        logger.info(f' robot motor id ordering: {self.motor_id_ordering} len:{len(self.motor_id_ordering)}')

        # NOTE: `active joint` ordering must be same as Mujoco `qpos`, see: mujoco_sim.py: def get_joint_state(self).
        self.active_joint_name_ordering = tuple(self.init_active_joint_angles)
        logger.info(f' robot active joint name ordering: {self.active_joint_name_ordering} len:{len(self.active_joint_name_ordering)}')

        # TODO: only transmission `ankle` will increase the len(active_joint_name_ordering) more than len(motor_name_ordering).
        # but we don't use `ankle` transmission till now. so, active_joint_name_ordering and motor_name_ordering are 1-to-1 mapping.
        assert len(self.motor_name_ordering) == len(self.motor_id_ordering) == len(self.active_joint_name_ordering)

        # self.nu = len(self.motor_name_ordering)

        # self.default_motor_angles = {}
        # for joint_name, joint_config in self.config["joints"].items():
        #     if not joint_config["is_passive"]:
        #         self.default_motor_angles[joint_name] = joint_config["default_pos"]
        #
        # self.default_active_joint_angles = self.motor_to_joint_angles(
        #     self.default_motor_angles
        # )

        # joints_config = self.config["joints"]
        # self.motor_to_active_joint_name = {}
        # self.active_joint_to_motor_name = {}

        for motor_name, joint_name in zip(self.motor_name_ordering, self.active_joint_name_ordering):
            transmission = self.config['joints'][motor_name]["transmission"]
            if transmission == "ankle":
                if "left" in motor_name:
                    self.motor_to_active_joint_name[motor_name] = [
                        "left_ank_roll",
                        "left_ank_pitch",
                    ]
                    self.active_joint_to_motor_name[joint_name] = [
                        "left_ank_act_1",
                        "left_ank_act_2",
                    ]
                elif "right" in motor_name:
                    self.motor_to_active_joint_name[motor_name] = [
                        "right_ank_roll",
                        "right_ank_pitch",
                    ]
                    self.active_joint_to_motor_name[joint_name] = [
                        "right_ank_act_1",
                        "right_ank_act_2",
                    ]
            elif transmission == "waist":
                self.motor_to_active_joint_name[motor_name] = ["waist_roll", "waist_yaw"]
                self.active_joint_to_motor_name[joint_name] = ["waist_act_1", "waist_act_2"]
            else:
                self.motor_to_active_joint_name[motor_name] = [joint_name]
                self.active_joint_to_motor_name[joint_name] = [motor_name]

        # self.passive_joint_names = []
        # TODO: here ,the `passive` is not same as `is_passive` in config.
        for joint_name in self.active_joint_name_ordering:
            transmission = self.config['joints'][joint_name]["transmission"]
            if transmission == "linkage":
                for suffix in [
                    "_front_rev_1",
                    "_front_rev_2",
                    "_back_rev_1",
                    "_back_rev_2",
                ]:
                    self.passive_joint_names.append(joint_name + suffix)
            elif transmission == "rack_and_pinion":
                self.passive_joint_names.append(joint_name + "_mirror")

        logger.info(f' robot passive joint names: {self.passive_joint_names}')

        # if "foot_name" in self.config["general"]:
        #     self.foot_name = self.config["general"]["foot_name"]

        # self.has_gripper = False
        for motor_name in self.motor_name_ordering:
            if "gripper" in motor_name:
                self.has_gripper = True

        # self.collider_names = []
        for link_name, link_config in self.collision_config.items():
            if link_config["has_collision"]:
                self.collider_names.append(link_name)

        # self.nu = len(self.motor_name_ordering)
        # self.joint_cfg_groups = {}
        # for joint_name, joint_config in self.config["joints"].items():
        #     self.joint_cfg_groups[joint_name] = joint_config["group"]

        # self.joint_cfg_limits = {}
        # for joint_name, joint_config in self.config["joints"].items():
        #     self.joint_cfg_limits[joint_name] = [
        #         joint_config["lower_limit"],
        #         joint_config["upper_limit"],
        #     ]

    # get attrs from all the joints in group.
    # NOTE: data is read from `config`, which is also dynamical through set_joint_config_attrs.
    def get_joint_config_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str = "name",
        group: str = "all",
    ) -> List[Any]:
        """Returns a list of attributes for joints that match specified criteria.

        Args:
            key_name (str): The key to search for in each joint's configuration.
            key_value (Any): The value that the specified key must have for a joint to be included.
            attr_name (str, optional): The attribute to retrieve from each matching joint. Defaults to "name".
            group (str, optional): The group to which the joint must belong. Use "all" to include joints from any group. Defaults to "all".

        Returns:
            List[Any]: A list of attributes from joints that match the specified key-value pair and group criteria.
        """
        attrs: List[Any] = []
        for joint_name, joint_config in self.config["joints"].items():
            if (
                key_name in joint_config
                and joint_config[key_name] == key_value
                and (joint_config["group"] == group or group == "all")
            ):
                if attr_name == "name":
                    attrs.append(joint_name)
                else:
                    attrs.append(joint_config[attr_name])

        return attrs

    # same order as motor_name_ordering.
    def get_motor_ordered_config_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str = "name",
        group: str = "all",
    ) -> List[Any]:
        """Returns a list of attributes for joints that match specified criteria.

        Args:
            key_name (str): The key to search for in each joint's configuration.
            key_value (Any): The value that the specified key must have for a joint to be included.
            attr_name (str, optional): The attribute to retrieve from each matching joint. Defaults to "name".
            group (str, optional): The group to which the joint must belong. Use "all" to include joints from any group. Defaults to "all".

        Returns:
            List[Any]: A list of attributes from joints that match the specified key-value pair and group criteria.
        """
        attrs: List[Any] = []
        # for joint_name, joint_config in self.config["joints"].items():
        for _name in self.motor_name_ordering:
            cfg = self.config['joints'][_name]
            if (
                key_name in cfg
                and cfg[key_name] == key_value
                and (cfg["group"] == group or group == "all")
            ):
                if attr_name == "name":
                    attrs.append(_name)
                else:
                    attrs.append(cfg[attr_name])

        # not guarantee len(attrs) == self.nu
        return attrs


    # set attrs for all the joints in group.
    # NOTE: support dynamically modify joint attrs, but modify ID and Name of motor is not allowed.
    def set_joint_config_attrs(
        self,
        key_name: str,
        key_value: Any,
        attr_name: str,
        attr_values: Mapping[int, float] | Sequence[float],
        group: str = "all",
    ):
        """Sets attributes for joints in the configuration based on specified criteria.

        Args:
            key_name (str): The key to match in each joint's configuration.
            key_value (Any): The value that the key must have for the joint to be modified.
            attr_name (str): The name of the attribute to set for matching joints.
            attr_values (Any): The values to assign to the attribute. Can be a dictionary or a list.
            group (str, optional): The group to which the joint must belong to be modified. Defaults to "all".
        """

        # TODO: add more attrs allowed for dynamically change.
        if attr_name not in {'init_pos',}:
            raise ValueError(f'not allowed to set joint attr: {attr_name} dynamically.')

        _index = 0
        for joint_name, joint_config in self.config["joints"].items():
            if (
                key_name in joint_config
                and joint_config[key_name] == key_value
                and (joint_config["group"] == group or group == "all")
            ):
                if isinstance(attr_values, Mapping):
                    # _id = joint_config["id"]
                    self.config["joints"][joint_name][attr_name] = attr_values[joint_config["id"]]
                else:
                    self.config["joints"][joint_name][attr_name] = attr_values[_index]
                    _index += 1

    @property
    def nu(self)->int:
        # read only.
        return len(self.motor_name_ordering)

    @property
    def foot_name(self)->str:
        name = ''
        if "foot_name" in self.config["general"]:
            name = self.config["general"]["foot_name"]
        return name

    @staticmethod
    def waist_fk(*, motor_pos: Dict[str, float|npt.NDArray[np.float32]], offsets: Mapping[str, float]) \
            -> Tuple[float|npt.NDArray[np.float32],float|npt.NDArray[np.float32]]:
        """Calculates the forward kinematics for the waist joint based on motor positions.

        Args:
            motor_pos (List[float]): A list containing the positions of the motors.
            offsets

        Returns:
            List[float]: A list containing the calculated waist roll and yaw angles.
        """
        # offsets = self.config["general"]["offsets"]
        # waist_roll = offsets["waist_roll_coef"] * (-motor_pos[0] + motor_pos[1])
        # waist_yaw = offsets["waist_yaw_coef"] * (motor_pos[0] + motor_pos[1])

        waist_roll = offsets["waist_roll_coef"] * (-motor_pos['waist_act_1'] + motor_pos['waist_act_2'])
        waist_yaw = offsets["waist_yaw_coef"] * (motor_pos['waist_act_1'] + motor_pos['waist_act_2'])

        return waist_roll, waist_yaw

    @staticmethod
    def waist_ik(*, offsets: Mapping[str, float], waist_pos: Dict[str, float|npt.NDArray[np.float32]]) \
            -> Tuple[float|npt.NDArray[np.float32], float|npt.NDArray[np.float32]]:
        """Calculates the inverse kinematics for the waist actuators based on the desired waist position.

        Args:
            waist_pos (List[float]): A list containing the desired roll and yaw positions of the waist.
            offsets

        Returns:
            List[float]: A list containing the calculated positions for the two waist actuators.
        """
        # offsets = self.config["general"]["offsets"]
        # roll = waist_pos[0] / offsets["waist_roll_coef"]
        # yaw = waist_pos[1] / offsets["waist_yaw_coef"]

        # NOTE:input joints must contain 'waist_roll' and 'waist_yaw'.
        roll = waist_pos['waist_roll'] / offsets["waist_roll_coef"]
        yaw = waist_pos['waist_yaw'] / offsets["waist_yaw_coef"]

        waist_act_1 = (-roll + yaw) / 2.
        waist_act_2 = (roll + yaw) / 2.
        return waist_act_1, waist_act_2

    def motor_to_active_joint_angles(self, #joints_config: Mapping[str, Any],
                                     motor_angles: OrderedDict[str, float|npt.NDArray[np.float32]],
                                     partial:bool = False )\
            -> OrderedDict[str, float|npt.NDArray[np.float32]]:
        """Converts motor angles to joint angles based on the robot's configuration.

        Args:
            motor_angles (OrderedDict[str, float]): A dictionary mapping motor names to their respective angles or angles of time sequence.
            partial(bool): allow input motor_angles contains only part of motors. defaults to False.

        Returns:
            OrderedDict[str, float]: A dictionary mapping joint names to their calculated angles.
        """

        if not partial:
            if set(motor_angles) != set(self.motor_name_ordering):
                raise ValueError(f'{motor_angles=:} not equals {self.motor_name_ordering=:}')

        # joint_angels: float: single angle; npt.NDArray(np.float32): a sequence of angels.
        joint_angles: OrderedDict[str, float|npt.NDArray[np.float32]] = OrdDictCls()

        # waist_act_pos: List[float|npt.NDArray[np.float32]] = []
        waist_act_pos: Dict[str, float|npt.NDArray[np.float32]] = {}

        # TODO: no use of ank ?
        left_ank_act_pos: List[float|npt.NDArray[np.float32]] = []
        right_ank_act_pos: List[float|npt.NDArray[np.float32]] = []

        # NOTE: only transmission `ankle` will increase the len(joint_angles) more than len(motor_angles).
        # but we don't use `ankle` transmission till now. so, len(motor_name) == len(active_joint_name).
        # joints_cfg :Dict[str, Any] = self.config['joints']
        for _m_name, _m_pos in motor_angles.items():
            transmission = self.config['joints'][_m_name]["transmission"]

            if transmission == "gear":
                # NOTE: a joint driven by gear, e.g., right_elbow_yaw_driven, even whose `is_passive` is true in config.json,
                # is still one of self.active_joint, and is not `self.passive_joint` which only include
                # `linkage` and `rack_and_pinion` transmission joints.

                joint_name = _m_name.replace("_drive", "_driven")
                joint_angles[joint_name] = (
                        - _m_pos * self.config['joints'][_m_name]["gear_ratio"]
                )

            elif transmission == "rack_and_pinion":
                joint_pinion_name = _m_name.replace("_rack", "_pinion")
                joint_angles[joint_pinion_name] = (
                        - _m_pos * self.config['joints'][_m_name]["gear_ratio"]
                )

            # input motors must contain 'waist_act_1' and 'waist_act_2'
            elif transmission == "waist":   # motors: `waist_act_1` and `waist_act_2`
                if not ('waist_act_1' in motor_angles
                        and 'waist_act_2' in motor_angles):
                    raise ValueError(
                        f'waist_act_1 and waist_act_2 must be provided together for converting from motor to active joint angles.'
                        f'input motor_angles keys: {motor_angles.keys()}')

                # Placeholder to ensure the correct order
                joint_angles["waist_roll"] = 0.0
                joint_angles["waist_yaw"] = 0.0
                # waist_act_pos.append(motor_pos)
                # guarantee the correct value of `waist_act_1` and `waist_act_2`.
                waist_act_pos[_m_name] = _m_pos

            elif transmission == "linkage":
                joint_angles[_m_name.replace("_act", "")] = _m_pos

            # only transmission `ankle` will increase the len(joint_angles) more than len(motor_angles).
            elif transmission == "ankle":
                if "left" in _m_name:
                    if not ('left_ank_act_1' in motor_angles
                            and 'left_ank_act_2' in motor_angles):
                        raise ValueError(
                            f'left_ank_act_1 and left_ank_act_2 must be provided together for converting from motor to active joint angles.'
                            f'input motor_angles keys: {motor_angles.keys()}')

                    joint_angles["left_ank_roll"] = 0.0
                    joint_angles["left_ank_pitch"] = 0.0

                    left_ank_act_pos.append(_m_pos)

                elif "right" in _m_name:
                    if not ('right_ank_act_1' in motor_angles
                            and 'right_ank_act_2' in motor_angles):
                        raise ValueError(
                            f'right_ank_act_1 and right_ank_act_2 must be provided together for converting from motor to active joint angles.'
                            f'input motor_angles keys: {motor_angles.keys()}')

                    joint_angles["right_ank_roll"] = 0.0
                    joint_angles["right_ank_pitch"] = 0.0

                    right_ank_act_pos.append(_m_pos)

            elif transmission == "none":
                # in this case, active joint has same name as motor.
                joint_angles[_m_name] = _m_pos

        if len(waist_act_pos) > 0:
            joint_angles["waist_roll"], joint_angles["waist_yaw"] = Robot.waist_fk(motor_pos=waist_act_pos,
                                                                                   offsets=self.config["general"]["offsets"])

        return joint_angles

    def active_joint_to_motor_angles(self, #joints_config: Mapping[str, Any],
                                     joint_angles: OrderedDict[str, float|npt.NDArray[np.float32]],
                                     partial:bool = False) -> OrderedDict[str, float|npt.NDArray[np.float32]]:
        """Converts joint angles to motor angles based on the transmission type specified in the configuration.

        Args:
            joint_angles (OrderedDict[str, float]): A dictionary mapping joint names to their respective angles or angles of time sequence.
            partial(bool): allow input joint_angles contains only part of active joints. defaults to False.

        Returns:
            OrderedDict[str, float]: A dictionary mapping motor names to their calculated angles.
        """

        if not partial:
            assert set(joint_angles) == set(self.active_joint_name_ordering)

        motor_angles: OrderedDict[str, float|npt.NDArray[np.float32]] = OrdDictCls()

        # waist_pos: List[float|npt.NDArray[np.float32]] = []
        waist_pos: Dict[str, float|npt.NDArray[np.float32]] = {}

        # TODO: no use of ank ?
        left_ankle_pos: List[float|npt.NDArray[np.float32]] = []
        right_ankle_pos: List[float|npt.NDArray[np.float32]] = []

        # joint_pos: float: single pos; npt.NDArray(np.float32): a sequence of pos.
        for _j_name, _j_pos in joint_angles.items():
            transmission = self.config['joints'][_j_name]["transmission"]
            if transmission == "gear":
                motor_name = _j_name.replace("_driven", "_drive")
                motor_angles[motor_name] = (
                        - _j_pos / self.config['joints'][motor_name]["gear_ratio"]
                )
            elif transmission == "rack_and_pinion":
                motor_name = _j_name.replace("_pinion", "_rack")
                motor_angles[motor_name] = (
                        - _j_pos / self.config['joints'][motor_name]["gear_ratio"]
                )
            elif transmission == "waist":
                #  input joints must contain 'waist_roll' and 'waist_yaw'.
                if not ('waist_roll' in joint_angles
                        and 'waist_yaw' in joint_angles):
                    raise ValueError(f'waist_roll and waist_yaw must be provided together for converting from active joint to motor angle.'
                                     f'input joint_angles keys: {joint_angles.keys()}')

                # Placeholder to ensure the correct order
                motor_angles["waist_act_1"] = 0.0
                motor_angles["waist_act_2"] = 0.0
                # waist_pos.append(joint_pos)
                waist_pos[_j_name] = _j_pos

            elif transmission == "linkage":
                motor_angles[_j_name + "_act"] = _j_pos
            elif transmission == "ankle":
                if "left" in _j_name:
                    if not ('left_ank_roll' in joint_angles
                            and 'left_ank_pitch' in joint_angles):
                        raise ValueError(f'left_ank_roll and left_ank_pitch must be provided together for converting.'
                                         f'input joint_angles keys: {joint_angles.keys()}')

                    motor_angles["left_ank_act_1"] = 0.0
                    motor_angles["left_ank_act_2"] = 0.0
                    left_ankle_pos.append(_j_pos)
                elif "right" in _j_name:
                    if not ('right_ank_roll' in joint_angles
                            and 'right_ank_pitch' in joint_angles):
                        raise ValueError(f'right_ank_roll and right_ank_pitch must be provided together for converting.'
                                         f'input joint_angles keys: {joint_angles.keys()}')
                    motor_angles["right_ank_act_1"] = 0.0
                    motor_angles["right_ank_act_2"] = 0.0
                    right_ankle_pos.append(_j_pos)

            elif transmission == "none":
                motor_angles[_j_name] = _j_pos

        if len(waist_pos) > 0:
            motor_angles["waist_act_1"], motor_angles["waist_act_2"] = Robot.waist_ik(waist_pos=waist_pos,
                                                                                      offsets=self.config["general"]["offsets"])

        return motor_angles

    def joint_to_passive_angles(self, #joints_config: Mapping[str, Any],
                                joint_angles: OrderedDict[str, float]) \
            -> OrderedDict[str, float]:
        """Converts joint angles to passive angles based on the transmission type.

        This function processes a dictionary of joint angles and converts them into passive angles using the transmission configuration specified in the object's configuration. It supports two types of transmissions: 'linkage' and 'rack_and_pinion'. For 'linkage' transmissions, it generates additional passive angles with specific suffixes, applying a sign change for knee joints. For 'rack_and_pinion' transmissions, it mirrors the joint angle.

        Args:
            joint_angles (OrderedDict[str, float]): A dictionary where keys are joint names and values are their respective angles.

        Returns:
            OrderedDict[str, float]: A dictionary of passive angles derived from the input joint angles.
        """
        passive_angles: OrderedDict[str, float] = OrdDictCls()
        for joint_name, joint_pos in joint_angles.items():
            transmission = self.config['joints'][joint_name]["transmission"]
            if transmission == "linkage":
                sign = 1 if "knee" in joint_name else -1
                for suffix in [
                    "_front_rev_1",
                    "_front_rev_2",
                    "_back_rev_1",
                    "_back_rev_2",
                ]:
                    passive_angles[joint_name + suffix] = sign * joint_pos
            elif transmission == "rack_and_pinion":
                passive_angles[joint_name + "_mirror"] = joint_pos

        return passive_angles

    def __str__(self):
        logger.info(f'robot {self.name} info :---------->')
        logger.info(f' robot motor name ordering: {self.motor_name_ordering}')
        logger.info(f' robot motor id ordering: {self.motor_id_ordering}')

        logger.info(f' robot active joint name ordering: {self.active_joint_name_ordering}')
        logger.info(f' robot passive joint names: {self.passive_joint_names}')



