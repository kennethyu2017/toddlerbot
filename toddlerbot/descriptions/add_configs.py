import argparse
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
from typing import Any, Dict, Set, Tuple
from collections import OrderedDict

import numpy as np


_DYNAMIXEL_DEFAULT_KP_CASEFOLD :Dict[str, float] = {'2xc430':2100.,
                                                    '2xl430': 2100.,
                                                    'xc330': 1500.,
                                                    'xc430': 2100.,
                                                    'xm430': 2700.,}

_FEITE_DEFAULT_KP_CASEFOLD :Dict[str, float] = {'sm40bl':32. }

def _build_general_config(urdf_root: ET.Element, robot_name: str)->OrderedDict[str, Any]:

    # general_cfg: OrderedDict[str, Any] = OrderedDict()

    # if macos, use 3
    if platform.system() == "Darwin":
        dynamixel_baud = 2
        feite_baud = 1
    else:
        dynamixel_baud = 2
        feite_baud = 1

    # general_config: Dict[str, Any] = {
    general_cfg: OrderedDict[str, Any] = OrderedDict({
        "is_fixed": True,
        "has_imu": False,

        # set in _build_robot_default_config()
        # "has_dynamixel": has_dynamixel,
        # 'has_feite': has_feite,

        "dynamixel_baudrate": dynamixel_baud * 1000000,
        # TODO:  use 1Mbps, or 115200 for feite
        "feitei_baudrate": feite_baud * 1000000,
        # "feitei_baudrate": feite_baud * 115200,  # default for SMS/STS series.,
        "solref": [0.004, 1],
    } )

    if "sysID" not in robot_name and "arms" not in robot_name:
        general_cfg["is_fixed"] = False
        general_cfg["has_imu"] = True
        if "gripper" in robot_name:
            general_cfg["ee_name"] = "gripper_connector"
        else:
            general_cfg["ee_name"] = "hand"
        general_cfg["foot_name"] = "ank_roll_link"
        general_cfg["offsets"] = {
            "torso_z": 0.33605,
            "default_torso_z": 0.3267,
            "foot_to_com_x": 0.0027,
            "foot_to_com_y": 0.0364,
            "hip_roll_to_pitch_z": 0.024,
            "hip_pitch_to_knee_z": 0.096,
            "knee_to_ank_pitch_z": 0.1,
            # "imu_x": 0.0282,
            # "imu_y": 0.0,
            # "imu_z": 0.105483,
            # "imu_zaxis": "-1 0 0",
        }

    # Define the URDF file path
    is_neck_closed_loop = False
    is_waist_closed_loop = False
    is_knee_closed_loop = False
    is_ankle_closed_loop = False

    for _jnt in urdf_root.findall("joint"):
        jnt_name = _jnt.get("name")
        jnt_type = _jnt.get("type")
        if jnt_name is None or jnt_type is None:
            continue

        if "neck" in jnt_name and "act" in jnt_name and "fixed" not in jnt_type:
            is_neck_closed_loop = True
        if "waist" in jnt_name and "act" in jnt_name and "fixed" not in jnt_type:
            is_waist_closed_loop = True
        if "knee" in jnt_name and "act" in jnt_name and "fixed" not in jnt_type:
            is_knee_closed_loop = True
        if "ank" in jnt_name and "act" in jnt_name and "fixed" not in jnt_type:
            is_ankle_closed_loop = True

    general_cfg["is_neck_closed_loop"] = is_neck_closed_loop
    general_cfg["is_waist_closed_loop"] = is_waist_closed_loop
    general_cfg["is_knee_closed_loop"] = is_knee_closed_loop
    general_cfg["is_ankle_closed_loop"] = is_ankle_closed_loop

    if is_waist_closed_loop:
        general_cfg["waist_roll_backlash"] = 0.03
        general_cfg["waist_yaw_backlash"] = 0.001
        general_cfg["offsets"]["waist_roll_coef"] = 0.29166667
        general_cfg["offsets"]["waist_yaw_coef"] = 0.20833333

    if is_ankle_closed_loop:
        general_cfg["ank_solimp_0"] = 0.9999
        general_cfg["ank_solref_0"] = 0.004
        general_cfg["offsets"]["ank_act_arm_y"] = 0.00582666
        general_cfg["offsets"]["ank_act_arm_r"] = 0.02
        general_cfg["offsets"]["ank_long_rod_len"] = 0.05900847
        general_cfg["offsets"]["ank_short_rod_len"] = 0.03951266
        general_cfg["offsets"]["ank_rev_r"] = 0.01

    return general_cfg


def _parse_joint_limit(joint: ET.Element)->Dict[str, float]:
    jnt_limit = joint.find("limit")
    if jnt_limit is None:
        raise ValueError(f"Joint {joint.get('name')} does not have a limit tag.")
    else:
        lower_limit = float(jnt_limit.get("lower", -np.pi))
        upper_limit = float(jnt_limit.get("upper", np.pi))

        return {"lower_limit": lower_limit,
                "upper_limit": upper_limit,}

def _parse_transmission(*,jnt_name: str,
                        is_neck_closed_loop:bool,
                        is_waist_closed_loop:bool,
                        is_knee_closed_loop:bool,
                        is_ankle_closed_loop,
                        )->Dict[str, str|bool]:
    is_passive = False
    transmission = "none"

    if "drive" in jnt_name:
        transmission = "gear"
        if "driven" in jnt_name:
            is_passive = True

    elif "neck" in jnt_name and is_neck_closed_loop:
        transmission = "linkage"
        if "act" not in jnt_name:
            is_passive = True

    elif "waist" in jnt_name and is_waist_closed_loop:
        transmission = "waist"
        if "act" not in jnt_name:
            is_passive = True

    elif "knee" in jnt_name and is_knee_closed_loop:
        transmission = "linkage"
        if "act" not in jnt_name:
            is_passive = True

    elif "ank" in jnt_name and is_ankle_closed_loop:
        transmission = "ankle"
        if "act" not in jnt_name:
            is_passive = True

    elif "gripper" in jnt_name:
        transmission = "rack_and_pinion"
        if "pinion" in jnt_name:
            is_passive = True

    return {
             "is_passive": is_passive,
             "transmission": transmission,
           }


def _parse_joint_group(jnt_name: str)->Dict[str,str]:
    # group = "none"
    group_keywords = {
        "neck": ["neck"],
        "waist": ["waist"],
        "arm": ["sho", "elbow", "wrist", "gripper"],
        "leg": ["hip", "knee", "ank"],
    }

    for _grp_name, _keywords in group_keywords.items():
        if any(_wd in jnt_name for _wd in _keywords):
            return {'group': _grp_name }

    return {'group': 'none'}


def _parse_dynamics(*, robot_name: str,
                    jnt_name: str,
                    is_passive: bool,
                    mtr_model: str,
                    motor_dynamics: OrderedDict[str, Dict[str, Any]],
                    passive_joint_dynamics: OrderedDict[str, Dict[str, Any]],
                    )->OrderedDict[str, str | float]:

    joint_dict_dyn_part: OrderedDict[str, str | float] = OrderedDict()

    if is_passive:
        # if jnt_name in joint_dyn_config:
        #     for param_name in joint_dyn_config[jnt_name]:
        #         joint_dict[param_name] = joint_dyn_config[jnt_name][param_name]

        # default to zero for passive joints if not specified in passive_joint_dynamics.
        if passive_joint_dynamics is not None and jnt_name in passive_joint_dynamics:
            joint_dict_dyn_part.update(passive_joint_dynamics[jnt_name])
        else:
            # default to zero.
            joint_dict_dyn_part.update({
                "damping": 0.0,
                "armature": 0.0,
                "frictionloss": 0.0,
                "tau_max": 0.0,
                "q_dot_tau_max": 0.0,
                "q_dot_max": 0.0   }
            )
    else:
        # For sysID task, the motor_dynamics is None .... keep the default zero in joint_dict_dyn_part....
        # for motoring joint, dynamics params come from sysID_dynamics.json of that motor model.
        if motor_dynamics is not None and mtr_model in motor_dynamics:
            joint_dict_dyn_part.update(motor_dynamics[mtr_model])

            # for param_name in motor_dynamics[mtr_model]:
            #     joint_dict_dyn_part[param_name] = motor_dynamics[mtr_model][param_name]

        elif "sysID" in robot_name:
            # default to zero.
            joint_dict_dyn_part.update({
                "damping": 0.0,
                "armature": 0.0,
                "frictionloss": 0.0,
                "tau_max": 0.0,
                "q_dot_tau_max": 0.0,
                "q_dot_max": 0.0  }
            )

            print(f'robot: {robot_name} is a sysID task, not have motor dynamics till now. '
                  f'set them to 0 temply, then generate sysID_dynamics.json by sysID_opt.py')

        else:
            raise ValueError(f'non-passive joint: {jnt_name} can not find dynamics. '
                             f'It must have corresponding dynamics.')

        # elif jnt_name in passive_joint_dynamics:
        #     joint_dict_dyn_part.update(passive_joint_dynamics[jnt_name])
        #
        #     # for param_name in passive_joint_dynamics[jnt_name]:
        #     #     joint_dict_dyn_part[param_name] = passive_joint_dynamics[jnt_name][param_name]

    return joint_dict_dyn_part


def _parse_motor_mapping(*, jnt_name: str,
                         motoring_joint_name_ordering: Tuple[str],
                         robot_name: str,
                         joint_motor_mapping: OrderedDict[str, Dict[str, Any]],
                         transmission: str)->OrderedDict[str, str|float]:

    joint_dict_mtr_part: OrderedDict[str, str|float] = OrderedDict()

    # if not is_passive:

    # motoring joint:
    # if jnt_name not in joint_motor_mapping:
    if jnt_name not in motoring_joint_name_ordering:
        raise ValueError(f"joint name: {jnt_name} not found in the joint motor mapping.")

    # only joint "is_passive": false can have 'id'.
    # toddlerbot id value scope: 0~29
    # toddlerbot_arm joints should start with id 16
    if "arms" in robot_name:
        start_id = 16
    else:
        start_id = 0

    # TODO: motoring joint id, should be same as robot.py motor_id_ordering....
    joint_dict_mtr_part["id"] = motoring_joint_name_ordering.index(jnt_name) + start_id

    mtr_param: Dict[str, float | str] = joint_motor_mapping[jnt_name]
    mtr_model: str = mtr_param["motor"]

    if mtr_model.casefold() in _DYNAMIXEL_DEFAULT_KP_CASEFOLD:
        joint_dict_mtr_part["type"] = "dynamixel"
        # TODO: repeat assigning same value...
        # config_dict["general"]['has_dynamixel'] = True

    elif mtr_model.casefold() in _FEITE_DEFAULT_KP_CASEFOLD:
        joint_dict_mtr_part["type"] = "feite"
        # config_dict["general"]['has_feite'] = True

    else:
        raise ValueError(f'motor name is neither dynamixel, nor feite:  {mtr_model} ')

    joint_dict_mtr_part["spec"] = mtr_model
    # joint_dict["control_mode"] = "extended_position"

    if "gripper" in jnt_name:
        assert joint_dict_mtr_part['type'] != 'feite', 'Feite actuator can not provide torque control mode.'
        joint_dict_mtr_part["control_mode"] = "current_based_position"

    elif joint_dict_mtr_part['type'] == 'dynamixel':
        joint_dict_mtr_part["control_mode"] = "extended_position"

    elif joint_dict_mtr_part['type'] == 'feite':
        # TODO: confirm actually using mode.
        joint_dict_mtr_part["control_mode"] = "position"

    else:
        raise ValueError(f'can not find accurate motor control mode: jnt_name: {jnt_name} '
                         f'joint motor type: {joint_dict_mtr_part["type"]} ')

    # joint_dict["control_mode"] = (
    #     "current_based_position"
    #     if "gripper" in jnt_name
    #     else "extended_position"
    # )
    joint_dict_mtr_part["init_pos"] = (
        mtr_param["init_pos"]
        if "init_pos" in mtr_param
        else 0.0
    )
    joint_dict_mtr_part["default_pos"] = (
        mtr_param["default_pos"]
        if "default_pos" in mtr_param
        else 0.0
    )
    joint_dict_mtr_part["kp_real"] = mtr_param["kp"]
    joint_dict_mtr_part["ki_real"] = mtr_param["ki"]
    joint_dict_mtr_part["kd_real"] = mtr_param["kd"]
    joint_dict_mtr_part["kff2_real"] = mtr_param["kff2"]
    joint_dict_mtr_part["kff1_real"] = mtr_param["kff1"]

    if joint_dict_mtr_part["type"] == "dynamixel":
        # kp value is the control TBL value of Dynamixel actuators,
        # i.e., 128 times the used kp value of PD controller.
        joint_dict_mtr_part["kp_sim"] = mtr_param["kp"] / 128.

    elif joint_dict_mtr_part["type"] == "feite":
        joint_dict_mtr_part["kp_sim"] = mtr_param["kp"]

    else:
        raise ValueError(f'motor type error: {joint_dict_mtr_part["type"]} ')

    joint_dict_mtr_part["kd_sim"] = 0.0

    if transmission == "gear" or transmission == "rack_and_pinion":
        if "gear_ratio" in mtr_param:
            joint_dict_mtr_part["gear_ratio"] = mtr_param["gear_ratio"]
        else:
            joint_dict_mtr_part["gear_ratio"] = 1.0

    return joint_dict_mtr_part


def _generate_robot_default_config(*,
                                   robot_name: str,
                                   urdf_root: ET.Element,
                                   # general_config: Dict[str, Any],
                                   joint_motor_mapping: OrderedDict[str, Dict[str, Any]],
                                   motor_dynamics:  OrderedDict[str, Dict[str, Any]],
                                   passive_joint_dynamics:  OrderedDict[str, Dict[str, Any]],
                                   # motor_config: Dict[str, Dict[str, Any]],
                                   # joint_dyn_config: Dict[str, Dict[str, float]],
                                   ) -> OrderedDict[str, OrderedDict[str, Any]] :
    """Generates a default configuration dictionary for a robot based on its URDF structure and provided configurations.

    Args:
        robot_name (str): The name of the robot, used to determine specific configurations.
        urdf_root (ET.Element): The root element of the robot's URDF XML structure.
        #general_config (Dict[str, Any]): General configuration settings for the robot.
        joint_motor_mapping:
        motor_dynamics:
        passive_joint_dynamics:

        # motor_config (Dict[str, Dict[str, Any]]): Configuration settings for each motor, including control parameters.
        # joint_dyn_config (Dict[str, Dict[str, float]]): Dynamic configuration settings for each joint, such as damping and friction.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the default configuration for the robot, including general settings and detailed joint configurations.
    """
    # config_dict: Dict[str, Dict[str, Any]] = {"general": general_config, "joints": {}}
    config_dict: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
        # general = general_config,
        # joints = OrderedDict() ,
        # )


    # joint with motor, should be same as robot.py motor_name_ordering....
    motoring_joint_name_ordering : Tuple[str] = tuple(joint_motor_mapping)

    config_dict['general'] = _build_general_config(urdf_root, robot_name)
    config_dict['joints'] = OrderedDict()

    for _jnt in urdf_root.findall("joint"):
        jnt_type = _jnt.get("type")
        if jnt_type is None or jnt_type == "fixed":
            continue

        jnt_name = _jnt.get("name")
        if jnt_name is None:
            continue

        joint_dict: OrderedDict[str,Any] = OrderedDict()

        joint_dict.update( _parse_transmission(jnt_name=jnt_name,
                                               is_neck_closed_loop= config_dict['general']['is_neck_closed_loop'],
                                               is_waist_closed_loop= config_dict['general']['is_waist_closed_loop'],
                                               is_knee_closed_loop= config_dict['general']['is_knee_closed_loop'],
                                               is_ankle_closed_loop= config_dict['general']['is_ankle_closed_loop'],
                                               ) )

        joint_dict.update( _parse_joint_group(jnt_name) )

        joint_dict.update( _parse_joint_limit(_jnt) )

        # # joint_dict = {
        #     "is_passive": is_passive,
        #     "transmission": transmission,
        #     "group": group,
        #     "lower_limit": lower_limit,
        #     "upper_limit": upper_limit,
        #
        #     # dynamics default to zero for passive joints and sysID joint.
        #     "damping": 0.0,
        #     "armature": 0.0,
        #     "frictionloss": 0.0,
        # }

        # TODO: joint driven by motoring joint. different meaning against passive joints in robot.py
        if not joint_dict['is_passive']:
            joint_dict.update( _parse_motor_mapping(jnt_name=jnt_name,
                                 motoring_joint_name_ordering=motoring_joint_name_ordering,
                                 robot_name=robot_name,
                                 joint_motor_mapping=joint_motor_mapping,
                                 transmission=joint_dict["transmission"] )
                                )
            # TODO: repeat assigning same value.
            if joint_dict['type'] == 'dynamixel':
                config_dict["general"]['has_dynamixel'] = True

            elif joint_dict['type'] == 'feite':
                config_dict["general"]['has_feite'] = True

            else:
                raise ValueError(f"motor type is neither dynamixel, nor feite:  {joint_dict['type']} ")



        joint_dict.update( _parse_dynamics(robot_name=robot_name,
                                           jnt_name=jnt_name,
                                           is_passive=joint_dict['is_passive'],
                                           mtr_model=joint_dict['spec'],
                                           motor_dynamics=motor_dynamics,
                                           passive_joint_dynamics=passive_joint_dynamics,
                                           )
                           )

        config_dict["joints"][jnt_name] = joint_dict

    # joints_list = list(config_dict["joints"].items())

    # Sort the list of joints first by id (if exists, otherwise use a large number) and then by name
    # sorted_joints_list = sorted(
    #     joints_list, key=lambda item: (item[1].get("id", float("inf")), item[0])
    # )
    sorted_joints_kv_pairs = sorted(
        config_dict["joints"].items(),
        key=lambda item: (item[1].get("id", float("inf")), item[0])
    )

    # Create a new ordered dictionary from the sorted list
    # config_dict["joints"] = dict(sorted_joints_list)
    config_dict["joints"] = OrderedDict(sorted_joints_kv_pairs)

    return config_dict


def _build_joint_motor_mapping(robot_cfg_dir: Path, robot_name: str  )->OrderedDict[str, Dict[str, float|str] ]:
    # TODO: This one needs to be ORDERED
    # joint_motor_mapping.json: to specify motor types/params for each URDF non-passive joint.
    # motor_cfg_file = robot_cfg_dir/"config_motors.json"
    jnt_mtr_mapping_file = robot_cfg_dir / "joint_motor_mapping.json"
    print(f'try to open {jnt_mtr_mapping_file} to load joint motor mapping.')
    joint_motor_mapping: OrderedDict[str,  Dict[str, float|str] ] | None = None

    if jnt_mtr_mapping_file.exists():
        print(f'open {jnt_mtr_mapping_file}.')
        with open(jnt_mtr_mapping_file, "r") as _f:
            joint_motor_mapping = json.load(_f, object_pairs_hook=OrderedDict)

    elif "sysID" in robot_name:
        print(f'can not find {jnt_mtr_mapping_file}, use default values for sysID.')
        mtr_model = str(robot_name.split("_")[-1])
        _default_kp: float = 0.
        if mtr_model.casefold() in _DYNAMIXEL_DEFAULT_KP_CASEFOLD:
            _default_kp = _DYNAMIXEL_DEFAULT_KP_CASEFOLD[mtr_model.casefold()]

        elif mtr_model.casefold() in _FEITE_DEFAULT_KP_CASEFOLD:
            _default_kp = _FEITE_DEFAULT_KP_CASEFOLD[mtr_model.casefold()]

        else:
            raise ValueError(f'motor model: {mtr_model} is neither dynamixel nor feite')

        joint_motor_mapping = OrderedDict(
            # joint_0 = {"motor": str(args.robot_name.split("_")[-1]), "init_pos": 0.0}
            joint_0={"motor": mtr_model,
                     "init_pos": 0.0,  # will be updated during calibrate_zero.
                     "default_pos": 0.0,
                     "kp": _default_kp,
                     "ki": 0.0,
                     "kd": 0.0,
                     "kff1": 0.0,
                     "kff2": 0.0,
                     }
        )
        # save as default for sysID only.
        with open(jnt_mtr_mapping_file, "wt") as _f:
            json.dump(joint_motor_mapping, _f, indent=4)

    else:
        raise ValueError(f"{jnt_mtr_mapping_file} not found!")

    print(f'joint motor mapping : {joint_motor_mapping}')

    return joint_motor_mapping


# only for non-sysID task.
def _build_joint_dynamics(robot_cfg_dir: Path, motor_model: Set[str])\
        ->Tuple[ OrderedDict[str, Dict[str, float]], OrderedDict[str, Dict[str, float]] ]:
    # joint_dyn_config: Dict[str, Dict[str, float]] = {}
    # if "sysID" not in robot_name:
        # motor_name_list = [
        #     str(motor_config["motor"]) for motor_config in motor_config.values()
        # ]
        # for motor_name in motor_name_list:

    # from sysID result:
    motor_dynamics: OrderedDict[str, Dict[str, float] ] = OrderedDict()
    # for _cfg in motor_config.values():
    for _model in motor_model:
        # mtr_model = str(_cfg["motor"])
        # sysID_dynamics.json for a motor is auto-generated by sysID_opt.py
        sysID_result_file = Path("toddlerbot") / 'descriptions' / f"sysID_{_model}" / "sysID_dynamics.json"
        if sysID_result_file.exists():
            with open(sysID_result_file, "r") as _f:
                _result = json.load(_f)
                motor_dynamics[_model] = _result["joint_0"]
        else:
            raise FileNotFoundError(f'can not find sysID result file: {sysID_result_file.resolve()}'
                                    f' for motor model: {_model}' )

        # motor_dynamics[_model] = sysID_result["joint_0"]

    # for sysID task, no passive_joint_dynamics.json required, so we don't need to load that file.
    # if 'sysID' not in robot_name:
        # several special passive driven joints, and the others passive joints have no dynamics params which using
        # default in mujoco.

    # TODO: the meaning of passive joint here, is different against which in robot.py.....
    # TODO: the meaning of passive joint here, is different against which in robot.py.....
    # TODO: the meaning of passive joint here, is different against which in robot.py.....
    # passive joint: driven by `motor` joint.
    passive_jnt_dyn_file: Path = robot_cfg_dir / "passive_joint_dynamics.json"
    passive_jnt_dynamics: OrderedDict[str, Dict[str, float] ] = OrderedDict()

    if passive_jnt_dyn_file.exists():
        with open(passive_jnt_dyn_file, "rt") as _f:
            passive_jnt_dynamics = json.load(_f, object_pairs_hook=OrderedDict)
        # joint_dyn_config.update(passive_jnt_dyn_config)
        # for _jnt_name, _cfg in passive_jnt_dyn_config.items():
        #     joint_dyn_config[_jnt_name] = _cfg

    return motor_dynamics, passive_jnt_dynamics


def _update_config_collision(robot_cfg_dir: Path, urdf_root: ET.Element):
    # config collision geom for URDF links.
    # can generate config_collision.json if not exist, and set not collision for all URDF links.
    collision_cfg_file = robot_cfg_dir / "config_collision.json"
    collision_config: OrderedDict[str, Any] | None = None

    if not collision_cfg_file.exists():
        collision_config = OrderedDict()
    else:
        with open(collision_cfg_file, "r") as _f:
            collision_config = json.load(_f, object_pairs_hook=OrderedDict)

    urdf_link_names: set[str] = {
        _lnk.get('name') for _lnk in urdf_root.findall('link')
        if _lnk.get('name') is not None
    }

    collision_cfg_updated: OrderedDict[str, Any] = OrderedDict()
    for _lnk_name in (urdf_link_names & set(collision_config)):
        collision_cfg_updated[_lnk_name] = collision_config[_lnk_name]

    for _lnk_name in (urdf_link_names - set(collision_config)):
        collision_cfg_updated[_lnk_name] = {"has_collision": False}

    # for link in root.findall("link"):
    #     link_name = link.get("name")
    #     if link_name is not None:
    #         link_names.append(link_name)

    #
    #     if link_name is not None and link_name not in collision_config:
    #         collision_config[link_name] = {"has_collision": False}

    # collision_config_updated = {
    #     _lnk_name: {"has_collision": False}
    #     if _lnk_name not in ( set(collision_config) & link_names)
    #     else collision_config[_lnk_name]
    #     for _lnk_name in link_names
    # }

    # collision_config_updated = OrderedDict(
    #     (_lnk_name, _v)
    #     for _lnk_name, _v in collision_config.items()
    #     if _lnk_name in link_names
    #     )

    with open(collision_cfg_file, "w") as _f:
        _f.write(json.dumps(collision_cfg_updated, indent=4))

    print(f"updated collision config file saved to {collision_cfg_file}")


def _main() -> None:
    """Main function to generate and save configuration files for a specified robot.

    This function parses command-line arguments to determine the robot's name and sets up
    various configuration parameters based on the robot's characteristics. It generates
    general, motor, joint dynamics, and collision configurations, saving them to JSON files
    in the appropriate directories.

    Raises:
        ValueError: If the motor configuration file is not found for the specified robot.
    """
    parser = argparse.ArgumentParser(description="Get the config.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    # baud is either 3 or 4 int
    # parser.add_argument(
    #     "--baud",
    #     type=int,
    #     default=4,
    #     help="The baudrate of motors, unit in Mbps",
    # )
    args = parser.parse_args()

    # has_dynamixel : bool = False
    # has_feite: bool = False
    # if "sysID" in args.robot_name:
    #     motor_model = str(args.robot_name.split("_")[-1]).casefold()
    #     if motor_model in _DYNAMIXEL_DEFAULT_KP_CASEFOLD:
    #         has_dynamixel = True
    #     elif motor_model in _FEITE_DEFAULT_KP_CASEFOLD:
    #         has_feite = True
    #     else:
    #         raise ValueError(f'motor name is neither dynamixel, nor feite:  {motor_model} ')

    # # if macos, use 3
    # if platform.system() == "Darwin":
    #     dynamixel_baud = 2
    #     feite_baud=1
    # else:
    #     dynamixel_baud = 2
    #     feite_baud = 1
    #
    # general_config: Dict[str, Any] = {
    #     "is_fixed": True,
    #     "has_imu": False,
    #
    #     # set in _build_robot_default_config()
    #     # "has_dynamixel": has_dynamixel,
    #     # 'has_feite': has_feite,
    #
    #     "dynamixel_baudrate": dynamixel_baud * 1000000,
    #     "feitei_baudrate": feite_baud * 1000000,
    #     "solref": [0.004, 1],
    # }
    #
    # if "sysID" not in args.robot_name and "arms" not in args.robot_name:
    #     general_config["is_fixed"] = False
    #     general_config["has_imu"] = True
    #     if "gripper" in args.robot_name:
    #         general_config["ee_name"] = "gripper_connector"
    #     else:
    #         general_config["ee_name"] = "hand"
    #     general_config["foot_name"] = "ank_roll_link"
    #     general_config["offsets"] = {
    #         "torso_z": 0.33605,
    #         "default_torso_z": 0.3267,
    #         "foot_to_com_x": 0.0027,
    #         "foot_to_com_y": 0.0364,
    #         "hip_roll_to_pitch_z": 0.024,
    #         "hip_pitch_to_knee_z": 0.096,
    #         "knee_to_ank_pitch_z": 0.1,
    #         # "imu_x": 0.0282,
    #         # "imu_y": 0.0,
    #         # "imu_z": 0.105483,
    #         # "imu_zaxis": "-1 0 0",
    #     }

    # if general_config["has_imu"]:
    #     imu_config_path = os.path.join(robot_dir, "config_imu.json")
    #     if os.path.exists(imu_config_path):
    #         with open(imu_config_path, "r") as f:
    #             general_config["_imu"] = json.load(f)
    #     else:
    #         raise ValueError(f"{imu_config_path} not found!")

    robot_cfg_dir: Path = Path("toddlerbot") / "descriptions" / args.robot_name
    urdf_file = robot_cfg_dir / f"{args.robot_name}.urdf"
    if not urdf_file.exists():
        raise FileNotFoundError(f'can not find urdf file: {urdf_file.resolve()}')

    urdf_tree: ET.ElementTree = ET.parse(urdf_file)
    urdf_root: ET.Element = urdf_tree.getroot()

    assert urdf_root.get('name') == args.robot_name, f'urdf root robot name: {urdf_root.get("name")} '

    # for joint with motor.
    joint_motor_mapping: OrderedDict[str, Dict[str, float|str] ] = _build_joint_motor_mapping(robot_cfg_dir, args.robot_name)

    # for sysID task, no passive_joint_dynamics.json required, so we don't need to load that file.
    motor_dynamics: OrderedDict[str, Any] |None = None
    passive_joint_dynamics: OrderedDict[str, Any]| None = None
    if "sysID" not in args.robot_name:
        _mtr_model: Set[str] = { _v['motor']  for _v in joint_motor_mapping.values() }
        motor_dynamics, passive_joint_dynamics = _build_joint_dynamics(robot_cfg_dir, _mtr_model)

    default_config = _generate_robot_default_config(
        robot_name= args.robot_name,
        urdf_root= urdf_root,
        # general_config = general_config,
        joint_motor_mapping=joint_motor_mapping,
        motor_dynamics=motor_dynamics,
        passive_joint_dynamics=passive_joint_dynamics,
        # motor_config= joint_motor_mapping,
        # joint_dyn_config= joint_dyn_config,
    )

    # save to config.json
    default_cfg_file = robot_cfg_dir/ "config.json"
    with open(default_cfg_file, "w") as _f:
        _f.write(json.dumps(default_config, indent=4))
        print(f"Config file saved to {default_cfg_file}")

    _update_config_collision(robot_cfg_dir, urdf_root)


if __name__ == "__main__":
    _main()
