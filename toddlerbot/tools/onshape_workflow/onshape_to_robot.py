"""Complete OnShape-to-robot conversion pipeline.

Orchestrates the full robot description generation workflow from OnShape CAD to
final XML and URDF files. Provides interactive prompts for each conversion step
and supports multiple robot configurations.
"""

# import argparse
import subprocess
import sys
from typing import NamedTuple,Tuple
from mujoco.viewer import launch_from_path
from ml_collections import config_dict

from toddlerbot.tools.onshape_workflow.get_xml import run_onshape_to_robot

def get_default_config() -> config_dict.FrozenConfigDict:
    """Return default configuration dictionary."""
    cfg = config_dict.create(
        robot_name='kbot_both_leg',
        onshape_to_robot_cfg = config_dict.create(
            merge_stl=True,
            simplify_stl = True,
            draw_frames=True,
            include_configuration_suffix=False,
            maxSTLSize=3,  # file size in MB.
        )
    )
    return config_dict.FrozenConfigDict(cfg)

class OnshapeID(NamedTuple):
    doc_id: str
    workspace_id: str

ASSEMBLY_ONSHAPE_ID_MAP = {
    'leg_L_subassembly': OnshapeID('f113281372e4169121004b65', 'c80b55f27aca32e7170d61fc'),
    'leg_R_subassembly': OnshapeID('f113281372e4169121004b65', 'c80b55f27aca32e7170d61fc'),
    'RS02_mount_box_with_long_arm_main_assembly': OnshapeID('ca7a3603bbd16cac2fa84c66', '16aeac7c0dcd110d2c86c4a7')

    # "2xc_430_palm": "565bc33af293a651f66e88d2",
    # "2xc_430_gripper": "565bc33af293a651f66e88d2",
    # "2xm_430_palm": "565bc33af293a651f66e88d2",
    # "2xm_430_gripper": "565bc33af293a651f66e88d2",
    # "teleop_leader": "565bc33af293a651f66e88d2",
    # "left_leg_2xc_430": "3084b13ad43394bd46cc00cf",
    # "right_leg_2xc_430": "3084b13ad43394bd46cc00cf",
    # "left_leg_2xm_430": "3084b13ad43394bd46cc00cf",
    # "right_leg_2xm_430": "3084b13ad43394bd46cc00cf",
    # "left_arm_palm": "322117012e09b07c7aec2a4a",
    # "right_arm_palm": "322117012e09b07c7aec2a4a",
    # "left_arm_gripper": "322117012e09b07c7aec2a4a",
    # "right_arm_gripper": "322117012e09b07c7aec2a4a",
    # "left_arm_leader": "322117012e09b07c7aec2a4a",
    # "right_arm_leader": "322117012e09b07c7aec2a4a",
    # "sysID_XC330": "1fb5d9a88ac086a053c4340b",
    # "sysID_XC430": "1fb5d9a88ac086a053c4340b",
    # "sysID_2XC430": "1fb5d9a88ac086a053c4340b",
    # "sysID_2XL430": "1fb5d9a88ac086a053c4340b",
    # "sysID_XM430": "1fb5d9a88ac086a053c4340b",

}

# names in onshape.
class RobotAssemblyNames(NamedTuple):
    body: Tuple[str,...]|None
    arm: Tuple[str,...]|None
    leg: Tuple[str,...]|None

ROBOT_ASSEMBLY = {
    'kbot_both_leg': RobotAssemblyNames(None, None, ('leg_L_subassembly', 'leg_R_subassembly') ),
    'sysID_device_RS02': RobotAssemblyNames(('RS02_mount_box_with_long_arm_main_assembly',  ), None, None),

    # "toddlerbot_2xc": {"body": "2xc_430_palm", "arm": "palm", "leg": "2xc_430"},
    # "toddlerbot_2xc_gripper": {
    #     "body": "2xc_430_gripper",
    #     "arm": "gripper",
    #     "leg": "2xc_430",
    # },
    # "toddlerbot_2xm": {"body": "2xm_430_palm", "arm": "palm", "leg": "2xm_430"},
    # "toddlerbot_2xm_gripper": {
    #     "body": "2xm_430_gripper",
    #     "arm": "gripper",
    #     "leg": "2xm_430",
    # },
    # "teleop_leader": {"body": "teleop_leader", "arm": "leader"},
    # "sysID_XC330": {"body": "sysID_XC330"},
    # "sysID_XC430": {"body": "sysID_XC430"},
    # "sysID_2XC430": {"body": "sysID_2XC430"},
    # "sysID_2XL430": {"body": "sysID_2XL430"},
    # "sysID_XM430": {"body": "sysID_XM430"},
}


def prompt_yes_no(prompt):
    """Prompts user for yes/no input and returns boolean result."""
    return input(f"{prompt} (y/n) > ").strip().lower() == "y"


def main():
    """Main entry point for OnShape-to-robot conversion pipeline."""
    # parser = argparse.ArgumentParser(
    #     description="Convert OnShape assemblies to URDF and MJCF"
    # )
    # parser.add_argument(
    #     "--robot",
    #     help="Robot name (e.g., toddlerbot_2xc)",
    #     default="",
    # )
    # parser.add_argument(
    #     "--assembly",
    #     nargs="*",
    #     help="Optional list of specific assemblies",
    #     default=None,
    # )
    # args = parser.parse_args()

    # robot = args.robot
    # assemblies = args.assembly
    cfg = get_default_config()

    if cfg.robot_name not in ROBOT_ASSEMBLY:
        raise ValueError(f"❌ Unknown robot name: {cfg.robot_name}")

    # body, arm, leg = ROBOT_ASSEMBLY[robot]
    assemblies = []
    for _a in ROBOT_ASSEMBLY[cfg.robot_name]:
        if _a is not None:
            assemblies.extend(_a)

    assert len(assemblies) > 0
    print(f'will search assemblies: {assemblies}')

    doc_ids = []
    workspace_ids = []
    for name in assemblies:
        if name not in ASSEMBLY_ONSHAPE_ID_MAP:
            raise ValueError(f"❌ assembly does not have onshape doc id : {name}")

        doc_ids.append(ASSEMBLY_ONSHAPE_ID_MAP[name].doc_id)
        workspace_ids.append(ASSEMBLY_ONSHAPE_ID_MAP[name].workspace_id)

    print(f'will download {len(doc_ids)} documents...')

    # repo = "toddlerbot"
    if prompt_yes_no("Do you want to export XML files from OnShape?"):
        scene_xml_files = run_onshape_to_robot(doc_ids,
                                               workspace_ids,
                                               assemblies,
                                               cfg.onshape_to_robot_cfg
                                               )
        for _f in scene_xml_files:
            if prompt_yes_no(f"Do you want to visualize the scene XML: {_f}"):
                launch_from_path(_f)

        # subprocess.run(
        #     [
        #         "bash",
        #         "-c",
        #         f"source ~/.bashrc && python3.11 {repo}/descriptions/get_xml.py "
        #         f"--doc-id-list {' '.join(doc_ids)} "
        #         f"--workspace-id-list {' '.join(workspace_ids)} "
        #         f"--assembly-list {' '.join(assemblies)}",
        #     ]
        # )
        print("\nExport completed.\n")
    else:
        print("\nExport skipped.\n")

    # if prompt_yes_no("Do you want to assemble the XML files?"):
    #     cmd = [
    #         "python",
    #         f"{repo}/descriptions/assemble_xml.py",
    #         "--robot",
    #         robot,
    #         "--torso-name",
    #         body,
    #     ]
    #     if arm:
    #         cmd += ["--arm-name", arm]
    #     if leg:
    #         cmd += ["--leg-name", leg]
    #     subprocess.run(cmd)
    #     print("\nAssembly completed.\n")
    # else:
    #     print("\nAssembly skipped.\n")

    # if prompt_yes_no("Do you want to visualize the XML?"):
    #     xml_path = f"{repo}/descriptions/{robot}/scene_pos_fixed.xml"
    #     subprocess.run(["python", "-m", "mujoco.viewer", "--mjcf", xml_path])

    # if prompt_yes_no("Do you want to convert the XML to URDF?"):
    #     subprocess.run(
    #         ["python", f"{repo}/descriptions/convert_to_urdf.py", "--robot", robot]
    #     )
    #     print("\nConversion to URDF completed.\n")
    #
    # if prompt_yes_no("Do you want to visualize the URDF?"):
    #     urdf_path = f"{repo}/descriptions/{robot}/{robot}.urdf"
    #     subprocess.run(["python", "-m", "mujoco.viewer", "--mjcf", urdf_path])




if __name__ == "__main__":
    main()
