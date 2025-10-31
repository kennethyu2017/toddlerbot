"""OnShape robot assembly XML generation utility.

Downloads and processes robot assemblies from OnShape CAD platform, converting them
to MuJoCo XML format for simulation. Handles configuration generation, mesh processing,
and file cleanup.
"""

import argparse
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import List
from ml_collections import config_dict

from toddlerbot.utils.io_utils import pretty_write_xml

def process_xml_and_stl_files(assembly_path: str, assembly_xml_name:str, stl_merged:bool=True) -> None:
    """Processes XML and STL files within a specified assembly directory.

    Args:
        assembly_path (str): The path to the directory containing the XML and STL files.
        assembly_xml_name (str): The name of the XML file.
        stl_merged (bool): Whether STL files are merged .

    Raises:
        ValueError: If no XML file is found in the specified directory.
    """
    xml_file = os.path.join(assembly_path, assembly_xml_name)
    if not os.path.exists(xml_file):
        raise ValueError("No XML file found in the robot directory.")

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    pretty_write_xml(root, xml_file)

    # Delete PART and unmerged STL files
    if stl_merged:
        assets_path = os.path.join(assembly_path, "assets")
        for entry in os.scandir(assets_path):
            if entry.is_file():
                os.remove(entry.path)
            elif entry.is_dir() and entry.name != "merged":
                shutil.rmtree(entry.path)


def run_onshape_to_robot(
        doc_ids: List[str],
        workspace_ids: List[str],
        assemblies: List[str],
        cfg: config_dict.FrozenConfigDict,
)->List[str]:
    """Downloads and converts OnShape assemblies to robot XML format.

    Args:
        doc_ids: List of OnShape document IDs
        workspace_ids: List of OnShape workspace IDs
        assemblies: List of assembly names to process
        cfg: onshape_to_robot configs.
    """
    parent_dir = os.path.join("toddlerbot", "descriptions", "assemblies")

    scene_xml_files = []
    # Process each assembly in series
    for _doc_id, _workspace_id, _assembly_name in zip(doc_ids, workspace_ids, assemblies):
        # under toddlerbot/descriptions/assemblies
        output_dir = os.path.join(parent_dir, _assembly_name)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)
        json_file_path = os.path.join(output_dir, "config.json")

        # joint_properties_dict = {}
        # equalities_dict = {}
        # if "leg" in assembly_name:
        #     base_assembly_name: str = "leg"
        #     config_name = assembly_name.replace("leg_", "")
        # elif "arm" in assembly_name:
        #     base_assembly_name = "arm"
        #     config_name = assembly_name.replace("arm_", "")
        #     for passive_joint in ["gripper_pinion", "gripper_pinion_mirror"]:
        #         joint_properties_dict[passive_joint] = {"actuated": False}
        # else:
        #     base_assembly_name = "toddlerbot"
        #     config_name = assembly_name
        #     for passive_joint in [
        #         "neck_pitch_front",
        #         "neck_pitch_back",
        #         "neck_pitch",
        #         "waist_roll",
        #         "waist_yaw",
        #     ]:
        #         joint_properties_dict[passive_joint] = {"actuated": False}
        #
        #     equalities_dict["closing_neck_pitch*"] = {
        #         "solref": "0.004 1",
        #         "solimp": "0.9999 0.9999 0.001 0.5 2",
        #     }

        json_data = {
            "document_id": _doc_id,
            'workspace_id': _workspace_id,
            "output_format": "mujoco",  # 'urdf',
            'output_filename': _assembly_name,  # basename only.
            "robot_name": _assembly_name, # cfg.robot_name,
            "assembly_name": _assembly_name,
            # "configuration": f"Configuration={config_name}",
            "include_configuration_suffix": cfg.include_configuration_suffix,
            "draw_frames": cfg.draw_frames,
            "merge_stls": cfg.merge_stl,
            "simplify_stls": cfg.simplify_stl,
            # TODO: kenneth: will reduce faces if beyond max stl size, so the collision calculation
            # will get trivial error ?
            "maxSTLSize": cfg.maxSTLSize,
            # "joint_properties": joint_properties_dict,
            # "equalities": equalities_dict,
        }

        # Write the JSON data to a file
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        # Execute the command
        subprocess.run(f"onshape-to-robot {output_dir}", shell=True)

        process_xml_and_stl_files(output_dir,
                                  f'{_assembly_name}.xml',
                                  stl_merged=cfg.merge_stl)

        scene_xml_files.append(os.path.join(output_dir, 'scene.xml'))

    return scene_xml_files

def main():
    """Main entry point for OnShape assembly processing."""
    parser = argparse.ArgumentParser(description="Process the xml.")
    parser.add_argument(
        "--doc-id-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the documents. Need to match the names in OnShape.",
    )
    parser.add_argument(
        "--assembly-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the assemblies. Need to match the names in OnShape.",
    )
    args = parser.parse_args()

    run_onshape_to_robot(args.doc_id_list, args.assembly_list)

if __name__ == "__main__":
    main()
