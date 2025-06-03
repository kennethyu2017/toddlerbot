import argparse
import json
# import os
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List

import numpy as np

from ..actuation import *
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports
# from toddlerbot.utils.misc_utils import log

from ._module_logger import logger


# This script is used to calibrate the zero points of the Dynamixel motors.


# TODO: impl.
def calibrate_feite():
    feite_config = FeiteConfig(
        # ????  init_pos= np.pi  ???
    )
    pass

def calibrate_dynamixel(port: str, robot: Robot, group: str)->Dict[int, float]:
    """Calibrates the Dynamixel motors for a specified group of joints in a robot.

    This function retrieves the necessary configuration and state information for the Dynamixel motors
    associated with a specified group of joints in the robot. It then calculates the initial positions
     for these motors and updates the robot's joint attributes accordingly.

    Args:
        port (str): The communication port used to connect to the Dynamixel motors.
        robot (Robot): The robot instance containing joint and configuration data.
        group (str): The group of joints to be calibrated.
    """
    dynamixel_ids = robot.get_joint_config_attrs("type", "dynamixel", "id", group)
    dynamixel_config = DynamixelConfig(
        port=port,
        baudrate=robot.config["general"]["dynamixel_baudrate"],
        control_mode=robot.get_joint_config_attrs("type", "dynamixel", "control_mode", group),
        kP=robot.get_joint_config_attrs("type", "dynamixel", "kp_real", group),
        kI=robot.get_joint_config_attrs("type", "dynamixel", "ki_real", group),
        kD=robot.get_joint_config_attrs("type", "dynamixel", "kd_real", group),
        kFF2=robot.get_joint_config_attrs("type", "dynamixel", "kff2_real", group),
        kFF1=robot.get_joint_config_attrs("type", "dynamixel", "kff1_real", group),
        # gear_ratio=robot.get_joint_attrs("type", "dynamixel", "gear_ratio", group),

        # init_pos=[],  # will be set to all-zero in controller.
        init_pos= None  # will be set to all-zero in controller.  #np.zeros_like(robot.motor_id_ordering, dtype=np.float32),  # set to all-zero.
    )

    # controller = DynamixelController(dynamixel_config, dynamixel_ids)
    # indexed by motor ID.
    motor_init_pos: Dict[int, float] = {}
    with DynamixelController.open_controller(dynamixel_config, dynamixel_ids) as controller:
        transmission_list = robot.get_joint_config_attrs(
            "type", "dynamixel", "transmission", group
        )
        joint_group_list = robot.get_joint_config_attrs("type", "dynamixel", "group", group)

        # TODO: after settle up the real robot, read out the angles of motors, as `init_pos` write into config.json.
        # then run_policy in real world, read out the `init_pos` from config.json and set to actuator_controller as
        # motor angle read/set offset.
        state_dict = controller.get_motor_state(retries=-1)

        assert len(transmission_list) == len(joint_group_list) == len(state_dict)

        for _trans, _group, (_id, _state) in zip(
            transmission_list, joint_group_list, state_dict.items()
        ):
            if _trans == "none" and _group == "arm":
                motor_init_pos[_id] = np.pi / 4 * round(_state.pos / (np.pi / 4))
            else:
                motor_init_pos[_id] = _state.pos

        # robot.set_joint_config_attrs("type", "dynamixel", "init_pos", init_pos, group)
        # controller.close_motors()

    return motor_init_pos


def main(args: argparse.Namespace): # robot: Robot, parts: List[str]):
    """Calibrates the robot's motors based on specified parts and updates the configuration.

    This function prompts the user to confirm the installation of calibration parts,
    calibrates the Dynamixel motors if present, and updates the motor configuration
    file with the initial positions of the specified parts.

    Args:
        args:
        # robot (Robot): The robot instance containing configuration and joint attributes.
        # parts (List[str]): A list of parts to calibrate. Can include specific parts
        #     like 'left_arm', 'right_arm', or 'all' to calibrate all parts.

    Raises:
        ValueError: If an invalid part is specified in the `parts` list.
        FileNotFoundError: If the motor configuration file is not found.
    """
    while True:
        response = input("Have you installed the calibration parts? (y/n) > ")
        response = response.strip().casefold()
        if response == "y" or response[0] == "y":
            break
        if response == "n" or response[0] == "n":
            return

        print("Please answer 'yes' or 'no'.")

    # Parse parts into a list
    parts = args.parts.split(" ") if args.parts != "all" else ["all"]
    robot = Robot(args.robot)

    executor = ThreadPoolExecutor(max_workers=5)

    has_dynamixel = robot.config["general"]["has_dynamixel"]

    future_dynamixel: Future|None = None

    TODO: Feite....

    if has_dynamixel:
        dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")

        # will modify robot.config .
        future_dynamixel:Future = executor.submit(
            calibrate_dynamixel, dynamixel_ports[0], robot, "all"
        )

    motor_init_pos: Dict[int, float] | None = None
    try:
        motor_init_pos = future_dynamixel.result()
    except Exception as err:
        logger.error(f'read init pos got an exception: {err}')
        raise
    else:
        logger.info(f'read init pos succeed: {motor_init_pos} ')
    finally:
        executor.shutdown(wait=True)

    #update robot config:
    assert len(motor_init_pos) == robot.nu
    robot.set_joint_config_attrs("type", "dynamixel", "init_pos", motor_init_pos, 'all')

    # Generate the motor mask based on the specified parts
    # only update motor init pos corresponding to `all_parts`, into joint_motor_mapping.json
    motor_ordering_idx_mask: set[int]
    # TODO: the number in list is ID or index of ordering? should be index, has value `0`.
    all_parts = {
        "left_arm": [16, 17, 18, 19, 20, 21, 22],
        "right_arm": [23, 24, 25, 26, 27, 28, 29],
        "left_gripper": [30],
        "right_gripper": [31],
        "hip": [2, 3, 4, 5, 6, 10, 11, 12],
        "knee": [7, 13],
        "left_ankle": [8, 9],
        "right_ankle": [14, 15],
        "neck": [0, 1],
    }
    if "all" in parts:
        motor_ordering_idx_mask = set(range(robot.nu))
    else:
        motor_ordering_idx_mask = set()

        assert  ( set(parts) & set(all_parts) ) == set(parts)
        # for _p in set(parts) & set(all_parts):
        for _p in parts:
            motor_ordering_idx_mask.update(all_parts[_p])

        # for _p in parts:
        #     if _p not in all_parts:
        #         raise ValueError(f"Invalid part: {_p}")
        #
        #     motor_mask.extend(all_parts[_p])

    # motor_names = robot.get_joint_config_attrs("is_passive", False)
    # robot.config modified in future_dynamixel.
    # TODO: should use motor id ordering to index motor init pos instead the original robot cfg...
    # motor_pos_init = np.array(robot.get_joint_config_attrs("is_passive", False, "init_pos"))

    # motor_angles = {}
    # for i, (name, pos) in enumerate(zip(motor_names, motor_init_pos)):
    #     if i in motor_id_mask:
    #         motor_angles[name] = round(pos, 4)

    # logger.info(f"Motor angles for selected parts: {motor_angles}")

    # motor_config_path = os.path.join(robot.root_path, "config_motors.json")
    motor_config_file = robot.root_path / 'joint_motor_mapping.json'
    if motor_config_file.exists():
        with open(motor_config_file, "rt") as f:
            motor_config = json.load(f)

        logger.info(f"update motor init pos for selected parts:")
        # for i, (name, pos) in enumerate(zip(motor_names, motor_init_pos)):
        # only update motor in mask.
        for _x, _id in enumerate(robot.motor_id_ordering):
            if _x in motor_ordering_idx_mask:
                motor_name = robot.id_to_motor_name[_id]
                pos = motor_init_pos[_id]
                motor_config[motor_name]["init_pos"] = pos
                logger.info(f'motor_name: {motor_name} , id:{_id} , ordering_idx:{_x}, pos: {pos} ')

        with open(motor_config_file, "wt") as f:
            json.dump(motor_config, f, indent=4)
    else:
        raise FileNotFoundError(f"Could not find {motor_config_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the zero point calibration.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="all",
        help="Specify parts to calibrate. Use 'all' or a subset of [left_arm, right_arm, left_gripper, right_gripper, hip, knee, left_ankle, right_ankle, neck], split by space.",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)

