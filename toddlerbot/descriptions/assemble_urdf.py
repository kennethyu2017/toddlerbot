import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List


@dataclass
class URDFConfig:
    """Data class for storing URDF configuration parameters."""

    robot_name: str
    body_name: str
    arm_name: str
    leg_name: str


def find_root_link_name(root: ET.Element):
    """Updates link names in `part_root` to ensure uniqueness in `body_root` and updates all references.

    Args:
        root (ET.Element): The root XML element of the part URDF being merged.

    Returns:
        str: The new unique name for the root link.
    """
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}  # type: ignore
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return str(root_link.pop())
    else:
        raise ValueError("Could not find root link in URDF")


def _update_link_names_and_references(body_root: ET.Element, part_root: ET.Element):
    """
    Updates link names in part_root to ensure uniqueness in body_root and updates all references.

    Args:
        body_root: The root XML element of the main URDF body.
        part_root: The root XML element of the part URDF being merged.

    Returns:
        None: The function directly modifies part_root in-place.
    """
    existing_links = {link.attrib["name"] for link in body_root.findall("link")}

    # Function to find or generate a unique link name
    def get_unique_name(old_name: str):
        if old_name not in existing_links:
            return old_name
        i = 2
        old_name_words = old_name.split("_")
        if old_name_words[-1].isdigit() and int(old_name_words[-1]) < 100:
            old_name = "_".join(old_name_words[:-1])

        while f"{old_name}_{i}" in existing_links:
            i += 1

        return f"{old_name}_{i}"

    # Update link names in part_root and collect changes
    name_changes = {}
    # find direct child.
    for link in part_root.findall("link"):
        old_name = link.attrib["name"]
        new_name = get_unique_name(old_name)
        if old_name != new_name:
            link.attrib["name"] = new_name
            name_changes[old_name] = new_name
            existing_links.add(new_name)

    for joint in part_root.findall("joint"):
        for tag in ["parent", "child"]:
            link_element = joint.find(tag)
            if link_element is not None and link_element.attrib["link"] in name_changes:
                link_element.attrib["link"] = name_changes[link_element.attrib["link"]]


def merge_part_into_body(*, body_root: ET.Element,
                         part_assembly_list: List[str],
                         assemblies_dir: Path,
                         urdf_config: URDFConfig)\
        -> ET.Element:
    for _jnt in body_root.findall("joint"):  # findall return list, not generator.
        child_link = _jnt.find("child")
        if child_link is None:
            continue

        child_link_name = child_link.get("link")
        if child_link_name is None:
            continue

        if not ("leg" in child_link_name and len(urdf_config.leg_name) > 0) and not (
                "arm" in child_link_name and len(urdf_config.arm_name) > 0
        ):
            continue

        for link in body_root.findall("link"):  # findall return list, not generator.
            # if link.get("name") == child_link_name.lower():
            if link.get("name") == child_link_name.casefold():
                body_root.remove(link)

        part_urdf_file: Path | None = None
        # part_assembly_name = ""

        # e.g., <child link="left_leg_active"/>
        # search corresponding urdf file of removed child link
        child_link_name_words = child_link_name.split("_")
        for _asm_name in part_assembly_list:
            name_words = _asm_name.split("_")
            if (
                    name_words[0].lower() == child_link_name_words[0].lower()
                    and name_words[1].lower() == child_link_name_words[1].lower()
            ):
                # NOTE: the `assembly_name` should be the part urdf file stem name,
                # see process_urdf_and_stl_files() in get_urdf.py
                part_urdf_file = assemblies_dir / _asm_name / (_asm_name + ".urdf")
                # part_assembly_name = _asm_name
                break

        if part_urdf_file is None:
            raise ValueError(f"Could not find part URDF for link '{child_link_name}'")

        part_tree = ET.parse(part_urdf_file)
        part_root = part_tree.getroot()

        _update_link_names_and_references(body_root, part_root)
        child_link.set("link", find_root_link_name(part_root))

        # append each direct child element, <link> and <joint>, in part_root into body_root, with
        # replaced mesh file path.
        for _element in list(part_root):
            # each element should be <link> or <joint>
            assert _element.tag in {'link', 'joint'}

            # Before appending, update the filename attribute in <mesh> tags
            # for mesh in element.findall(".//mesh"):
            for mesh in _element.iter("mesh"):
                mesh.attrib["filename"] = mesh.attrib["filename"].replace(
                    # "package:///", f"../assemblies/{part_assembly_name}/"
                    "package:///", f"../assemblies/{part_urdf_file.stem}/"
                )

            body_root.append(_element)

    return body_root

def assemble_urdf(urdf_config: URDFConfig):
    """Assembles a URDF file for a robot based on the provided configuration.

    This function constructs a complete URDF (Unified Robot Description Format) file by combining a base body URDF
    with optional arm and leg components specified in the configuration.
    It updates mesh file paths and ensures the correct structure for simulation.

    Args:
        urdf_config (URDFConfig): Configuration object containing the names of the robot, body, arms, and legs to be assembled.

    Raises:
        ValueError: If a source URDF for a specified link cannot be found.
    """
    # Parse the target URDF
    description_dir = Path("toddlerbot") / "descriptions"
    assemblies_dir = description_dir / "assemblies"

    # NOTE: body_name should be the assembly name under `descriptions/assemblies`.
    # see process_urdf_and_stl_files() in get_urdf.py
    # NOTE: the `assembly_name` should be the body urdf file stem name.
    body_urdf_file = assemblies_dir / urdf_config.body_name / (urdf_config.body_name + ".urdf")

    body_tree = ET.parse(body_urdf_file)
    body_root = body_tree.getroot()
    body_root.set("name", urdf_config.robot_name)

    # for sysID robot, we only specify body, same as robot_name, but without arm/leg, so as to
    # replace mesh file path and add mujoco tags.
    part_assembly_list: List[str] = []
    if len(urdf_config.arm_name) > 0:
        part_assembly_list.append("left_" + urdf_config.arm_name)
        part_assembly_list.append("right_" + urdf_config.arm_name)

    if len(urdf_config.leg_name) > 0:
        part_assembly_list.append("left_" + urdf_config.leg_name)
        part_assembly_list.append("right_" + urdf_config.leg_name)

    # only the direct child elements of body root. not recursively walk to all sub-elements.
    # for _element in list(body_root):
    # Before appending, update the filename attribute in <mesh> tags
    # for mesh in element.findall(".//mesh"):
    # for _mesh in _element.iter("mesh"):
    # TODO:  findall return a list, iter return a generator.
    for _mesh in body_root.iter("mesh"):
        _mesh.attrib["filename"] = _mesh.attrib["filename"].replace(
            # "package:///", f"../assemblies/{urdf_config.body_name}/"
            "package:///", f"../assemblies/{body_urdf_file.stem }/"
        )

    # only find direct child elements of body_root.not recursively walk to all sub-elements.
    # for sysID robot, we only specify body, without arm/leg.
    if len(part_assembly_list) > 0:
        body_root = merge_part_into_body(body_root=body_root,
                                         part_assembly_list=part_assembly_list,
                                         assemblies_dir=assemblies_dir,
                                         urdf_config=urdf_config
                                         )

    # Check if the <mujoco> element already exists
    mujoco = body_root.find("./mujoco")
    if mujoco is None:
        # Create and insert the <mujoco> element
        mujoco = ET.Element("mujoco")
        compiler = ET.SubElement(mujoco, "compiler")
        compiler.set("strippath", "false")
        compiler.set("balanceinertia", "true")
        compiler.set("discardvisual", "false")
        body_root.insert(0, mujoco)

    target_robot_dir = description_dir / urdf_config.robot_name
    # os.makedirs(target_robot_dir, exist_ok=True)
    target_robot_dir.mkdir(exist_ok=True)

    target_urdf_file :Path = target_robot_dir/ (urdf_config.robot_name + ".urdf")

    # for pretty format.
    ET.indent(body_tree)
    body_tree.write(target_urdf_file, encoding='utf-8')


def main():
    """Parses command-line arguments to configure and assemble a URDF (Unified Robot Description Format) file.

    This function sets up an argument parser to accept parameters for robot configuration, including the robot's name, body, arm, and leg components. It then calls the `assemble_urdf` function with the parsed arguments to generate the URDF file.
    """
    parser = argparse.ArgumentParser(description="Assemble the urdf.")
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        # default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--body-name",
        type=str,
        required=True,
        # default="4R_body",
        help="The name of the body.",
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="",
        help="The name of the arm.",
    )
    parser.add_argument(
        "--leg-name",
        type=str,
        default="",
        help="The name of the leg.",
    )
    args = parser.parse_args()

    assemble_urdf(URDFConfig(args.robot_name, args.body_name, args.arm_name, args.leg_name))


if __name__ == "__main__":
    main()
