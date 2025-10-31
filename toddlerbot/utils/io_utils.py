"""Input/output utilities for file operations and system interactions.

Provides functionality for XML formatting, serial port detection, file discovery,
and environment path resolution.
"""

import os
import platform
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional
from xml.dom.minidom import parseString

import serial.tools.list_ports as list_ports


def pretty_write_xml(root: ET.Element, file_path: str):
    """Formats an XML Element into a pretty-printed XML string and writes it to a specified file.

    Args:
        root (ET.Element): The root element of the XML tree to be formatted.
        file_path (str): The path to the file where the formatted XML will be written.
    """
    # Convert the Element or ElementTree to a string
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")

    # Parse and pretty-print the XML string
    dom = parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove blank lines
    pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

    # Write the pretty XML to the file
    with open(file_path, "w") as file:
        file.write(pretty_xml)


def find_ports(target: str) -> List[str]:
    """Find open network ports on a specified target.

    Args:
        target: The IP address or hostname of the target to scan for open ports.

    Returns:
        A list of strings representing the open ports on the target.
    """
    ports = list(list_ports.comports())
    target_ports: List[str] = []

    os_type = platform.system()

    for port, desc, hwid in ports:
        # Adjust the condition below according to your board's unique identifier or pattern
        print(port, desc, hwid)
        if target in desc:
            if os_type != "Windows":
                port = port.replace("cu", "tty")

            print(f"Found {target} board: {port} - {desc} - {hwid}")
            target_ports.append(port)

    if len(target_ports) == 0:
        raise ConnectionError(f"Could not find the {target} board.")
    else:
        return sorted(target_ports)


def find_last_result_dir(result_dir: str, prefix: str = "") -> Optional[str]:
    """
    Find the latest (most recent) result directory within a given directory.

    Args:
        result_dir: The path to the directory containing result subdirectories.
        prefix: The prefix of result directory names to consider.

    Returns:
        The path to the latest result directory, or None if no matching directory is found.
    """
    # Get a list of all items in the result directory
    try:
        dir_contents = os.listdir(result_dir)
    except FileNotFoundError:
        print(f"The directory {result_dir} was not found.")
        return None

    # Filter out directories that start with the specified prefix
    result_dirs = [
        d
        for d in dir_contents
        if os.path.isdir(os.path.join(result_dir, d)) and d.startswith(prefix)
    ]

    # Sort the directories based on name, assuming the naming convention includes a sortable date and time
    result_dirs.sort()

    # Return the last directory in the sorted list, if any
    if result_dirs:
        return os.path.join(result_dir, result_dirs[-1])
    else:
        print(f"No directories starting with '{prefix}' were found in {result_dir}.")
        return None


def find_latest_file_with_time_str(directory: str, file_prefix: str = "") -> str | None:
    """
    Finds the file with the latest timestamp (YYYYMMDD_HHMMSS) in the given directory,
    for files ending with the specified suffix.

    Args:
        directory (str): Directory to search for files.
        file_suffix (str): The suffix to match (e.g., '.pkl', '_updated.pkl').

    Returns:
        str | None: Full path of the latest file or None if no matching file is found.
    """
    pattern = re.compile(r".*" + re.escape(file_prefix) + r".*(\d{8}_\d{6}).*")

    latest_file = None
    latest_time = None

    # Iterate through files in the directory
    for file in os.listdir(directory):
        match = pattern.search(file)  # Check if the file matches the pattern
        if match:
            # Extract the timestamp and parse it into a datetime object
            timestamp_str = match.group(1)
            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            # Update the latest file if this timestamp is more recent
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file

    return os.path.join(directory, latest_file) if latest_file else None


def get_conda_path():
    """Determines the path of the current Python environment.

    Returns:
        str: The path to the current Python environment. If a virtual environment is active, returns the virtual environment's path. If a conda environment is active, returns the conda environment's path. Otherwise, returns the system environment's path.
    """
    # Check if using a virtual environment
    if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        prefix = sys.prefix
    # If using conda, the CONDA_PREFIX environment variable is set
    elif "CONDA_PREFIX" in os.environ:
        prefix = os.environ["CONDA_PREFIX"]
    else:
        # If not using virtualenv or conda, assume system environment
        prefix = sys.prefix

    if platform.system() == "Windows":
        env_path = os.path.join(prefix, "lib", "site-packages")
    else:
        env_path = os.path.join(
            prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )

    return env_path
