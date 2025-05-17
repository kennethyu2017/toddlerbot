"""
experimented for feite sm40bl actuator.  according to feite sms&sts_memory_table_220328.
by kenneth yu.
"""

import time
from threading import Lock
from typing import Dict, List, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from scservo_sdk import (SMS_STS_DEFAULT_BAUD_RATE, SMS_STS_EEPROM_Table_RW)
from . import BaseController, JointState
from .feite_client import FeiteGroupClient, PosVelLoadRecord

from ._module_logger import logger

CONTROL_MODE_DICT: Dict[str, int] = {
    "position": 0,
    "velocity": 1,
    "pwm": 2,  # open_loop
    "step": 3,
}

# TODO: move into yaml config file.
USB_SERIAL_DEFAULT_LATENCY_TIMER_IN_MS = 1

# def get_env_path():
#     """Determines the path of the current Python environment.
#
#     Returns:
#         str: The path to the current Python environment. If a virtual environment is active, returns the virtual environment's path. If a conda environment is active, returns the conda environment's path. Otherwise, returns the system environment's path.
#     """
#     # Check if using a virtual environment
#     if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
#         return sys.prefix
#     # If using conda, the CONDA_PREFIX environment variable is set
#     elif "CONDA_PREFIX" in os.environ:
#         return os.environ["CONDA_PREFIX"]
#     else:
#         # If not using virtualenv or conda, assume system environment
#         return sys.prefix


# on linux, default usb-serial latency timer usally is 16ms.
# def set_latency_timer(latency_value: int = 1):
#     """Sets the LATENCY_TIMER variable in the port_handler.py file to the specified value.
#
#     This function locates the port_handler.py file within the dynamixel_sdk package and updates the LATENCY_TIMER variable to the given `latency_value`. If the file or the variable is not found, an error message is printed.
#
#     Args:
#         latency_value (int): The value to set for LATENCY_TIMER. Defaults to 1.
#     """
#     # env_path = get_env_path()
#
#     # Construct the path to port_handler.py
#     port_handler_path = os.path.join(
#         env_path,
#         "lib",
#         f"python{sys.version_info.major}.{sys.version_info.minor}",
#         "site-packages",
#         "dynamixel_sdk",
#         "port_handler.py",
#     )
#
#     if not os.path.exists(port_handler_path):
#         print(f"Error: port_handler.py not found at {port_handler_path}")
#         return
#
#     try:
#         # Read the content of port_handler.py
#         with open(port_handler_path, "r") as file:
#             lines = file.readlines()
#
#         # Search for the LATENCY_TIMER line and modify it
#         modified = False
#         for i, line in enumerate(lines):
#             if "LATENCY_TIMER" in line:
#                 lines[i] = f"LATENCY_TIMER = {latency_value}\n"
#                 modified = True
#                 break
#
#         if modified:
#             # Write the modified content back to port_handler.py
#             with open(port_handler_path, "w") as file:
#                 file.writelines(lines)
#
#             print(f"LATENCY_TIMER set to 1 in {port_handler_path}")
#         else:
#             print("LATENCY_TIMER variable not found in port_handler.py")
#
#     except Exception as e:
#         print(f"Error while modifying the file: {e}")


class FeiteConfig(NamedTuple):
    """Data class for storing Feite SM40BL configuration parameters."""

    port: str = ''
    baudrate: int = SMS_STS_DEFAULT_BAUD_RATE
    control_mode: List[str] = []
    kP: List[int] = []    # only 1 byte for Feite KP, KD, KI.
    kI: List[int] = []
    kD: List[int] = []
    # kFF2: List[float]
    # kFF1: List[float]
    init_pos: List[float] = []
    default_vel: float = np.pi
    # interp_method: str = "cubic"
    return_delay_time: int = 1


class FeiteController(BaseController):
    """Class for controlling Feite SM40BL motors."""

    def __init__(self, config: FeiteConfig, motor_ids: List[int]):
        """Initializes the motor controller with the given configuration and motor IDs.

        Args:
            config (DynamixelConfig): The configuration settings for the Feite motors.
            motor_ids (List[int]): A list of motor IDs to be controlled.

        Attributes:
            config (DynamixelConfig): Stores the configuration settings.
            motor_ids (List[int]): Stores the list of motor IDs.
            lock (Lock): A threading lock to ensure thread-safe operations.
            init_pos (np.ndarray): An array of initial positions for the motors, initialized to zeros if not provided in the config.
        """
        client: FeiteGroupClient
        motor_ids: List[int]

        self.config = config
        self.motor_ids: List[int] = motor_ids
        self.lock = Lock()

        self.client =self.connect_to_client()
        self.initialize_motors()

        if len(self.config.init_pos) == 0:
            self.init_pos = np.zeros(len(motor_ids), dtype=np.float32)
        else:
            self.init_pos = np.array(config.init_pos, dtype=np.float32)
            self.update_init_pos()

    def connect_to_client(self, latency_value: int = 1)->FeiteGroupClient:
        """Connects to a Feite client and sets the USB latency timer.

        This method sets the USB latency timer for the specified port and attempts to connect to a Feite client.
        The latency timer is set differently based on the operating system.
         If the connection fails, an error is logged or raised.

        Args:
            latency_value (int): The desired latency timer value. Defaults to 1.

        Raises:
            ConnectionError: If the connection to the Feite port fails.
        """
        # os_type = platform.system()
        # try:
            # change the port_handler.py in dynamixel sdk.
            # set_latency_timer(latency_value)
            # TODO: can not write into CH340.
            # if os_type == "Linux":
            #     # Construct the command to set the latency timer on Linux
            #     command = f"echo {latency_value} | sudo tee /sys/bus/usb-serial/devices/{self.config.port.split('/')[-1]}/latency_timer"
            # elif os_type == "Darwin":
            #     command = f"./toddlerbot/actuation/latency_timer_setter_macOS/set_latency_timer -l {latency_value}"
            # else:
            #     raise Exception()

            # Run the command
            # result = subprocess.run(
            #     command, shell=True, text=True, check=True, stdout=subprocess.PIPE
            # )
            # logger.info(f"Latency Timer set: {result.stdout.strip()}", header="Feite")

        # except Exception as e:
        #     if os_type == "Windows":
        #         logger.error(
        #             "Make sure you're set the latency in the device manager!",
        #             header="Feite",
        #             level="warning",
        #         )
        #     else:
        #         logger.error(
        #             f"Failed to set latency timer: {e}",
        #             header="Feite",
        #             level="error",
        #         )

        # time.sleep(0.1)

        try:
            client = FeiteGroupClient(
                motor_ids = self.motor_ids,
                port_name = self.config.port,
                baud_rate= self.config.baudrate,)
            client.connect()
            logger.info(f"Connected to the port: {self.config.port}")
            return client

        except Exception:
            raise ConnectionError("Could not connect to the Feite port.")

    def initialize_motors(self):
        """Initialize the motors by rebooting, checking voltage, and configuring settings.

        This method performs the following steps:
        1. Reboots the motors.
        2. Checks the input voltage to ensure it is above a safe threshold.
        3. Configures various motor settings such as return delay time, control mode, and PID gains.
        4. Enables torque on the motors.

        Raises:
            ValueError: If the input voltage is below 10V, indicating a potential power supply issue.
        """
        logger.info("Initializing motors...")
        #TODO: Feite has no reboot function.
        # self.client.reboot(self.motor_ids)
        time.sleep(0.2)

        _, v_in = self.client.read_vin()
        assert len(v_in)==len(self.motor_ids)
        logger.info(f"Voltage (V): {v_in}")
        if np.any(v_in < 10):
            raise ValueError(
                "Voltage too low. Please check the power supply or charge the batteries."
            )

        time.sleep(0.2)
        # ---- TODO: add overload protect, min/max pos.. to Feite motors. ----

        # This sync writing section has to go after the voltage reading to make sure the motors are powered up
        # Set the return delay time to 1*2=2us
        self.client.sync_write_1_byte(motor_ids=self.motor_ids,
                                     params= [bytes([self.config.return_delay_time])] * len(self.motor_ids),
                                     address=SMS_STS_EEPROM_Table_RW.RETURN_DELAY_TIME)

        self.client.sync_write_1_byte(motor_ids=self.motor_ids,
                                     params= [bytes([CONTROL_MODE_DICT[m]]) for m in self.config.control_mode ],
                                     address=SMS_STS_EEPROM_Table_RW.MODE)

        # write kP,kD,kI together
        assert np.all(np.array(self.config.kP) < 0xff) and np.all(0 <= np.array(self.config.kP))
        assert np.all(np.array(self.config.kD) < 0xff) and np.all(0 <= np.array(self.config.kD))
        assert np.all(np.array(self.config.kI) < 0xff) and np.all(0 <= np.array(self.config.kI))

        self.client.sync_write_kP_kD_kI.....

        self.client.sync_write_3_bytes(motor_ids=self.motor_ids,
                                     params=[ bytes([_p, _d, _i]) for _p, _d, _i in
                                        zip(self.config.kP, self.config.kD, self.config.kI) ],
                                     address=SMS_STS_EEPROM_Table_RW.KP)

        # self.client.sync_write(motor_ids=self.motor_ids,
        #                        params=(bytes([_v]) for _v in self.config.kP),
        #                        address=SMS_STS_EEPROM_Table_RW.KP,
        #                        size=1)
        # self.client.sync_write(motor_ids=self.motor_ids,
        #                        params=(bytes([_v]) for _v in self.config.kD),
        #                        address=SMS_STS_EEPROM_Table_RW.KD,
        #                        size=1)
        # self.client.sync_write(motor_ids=self.motor_ids,
        #                        params=(bytes([_v]) for _v in self.config.kI),
        #                        address=SMS_STS_EEPROM_Table_RW.KI,
        #                        size=1)

        # TODO: no feedforward of Feite actuator.
        # self.client.sync_write(self.motor_ids, self.config.kFF2, 88, 2)
        # self.client.sync_write(self.motor_ids, self.config.kFF1, 90, 2)
        # self.client.sync_write(self.motor_ids, self.config.current_limit, 102, 2)

        self.client.set_torque_enabled(self.motor_ids, True)

        time.sleep(0.2)

    def update_init_pos(self):
        """Update the initial position to account for any changes in position.

        This method reads the current position from the client and calculates the
        difference from the stored initial position. It then adjusts the initial
        position to reflect any changes, ensuring that the position remains within
        the range of [-π, π].
        """
        _, pos_arr = self.client.read_pos(retries=-1)
        delta_pos = pos_arr - self.init_pos
        delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi
        self.init_pos = pos_arr - delta_pos

    def close_motors(self):
        """Closes all active motor clients.

        This method iterates over all currently open Feite clients and forces them to close if they are in use. It logs a message for each client that is being forcibly closed and then sets the client's port handler to not in use before disconnecting the client.
        """
        open_clients: Set[FeiteGroupClient] = FeiteGroupClient.OPEN_CLIENTS  # type: ignore
        for _client in open_clients:
            if _client.port_handler.is_using:
                logger.info("Forcing client to close.")
            _client.port_handler.is_using = False
            _client.disconnect()

    # Only enable/disable the torque, but stay connected through comm. If no id is provided, disable all motors
    @staticmethod
    def set_motor_torque(*, enabled:bool, ids:Sequence[int]=None):
        """Disables the torque for specified motors or all motors if no IDs are provided.

        Args:
            ids (list, optional): A list of motor IDs to disable. If None, all motors will be disabled.
            enabled(bool):
        """
        open_clients: Set[DynamixelClient] = FeiteGroupClient.OPEN_CLIENTS # type: ignore
        for _client in open_clients:
            if ids is not None:
                # get the intersecting list between ids and motor_ids
                set_ids = set(_client.motor_ids) & set(ids)
                logger.info(f"set motor id: {set_ids}")
            else:
                set_ids = _client.motor_ids
                logger.info(f"set all the motors in client with ids: {set_ids}")

            _client.set_torque_enabled(set_ids, enabled, retries=0)


    # def enable_motors(self, ids:Sequence[int]=None):
    #     """Enables torque for specified motors or all motors if no IDs are provided.
    #
    #     Args:
    #         ids (list, optional): A list of motor IDs to enable. If None, all motors will be enabled.
    #     """
    #     open_clients: Set[DynamixelClient] = FeiteGroupClient.OPEN_CLIENTS  # type: ignore
    #     for _client in open_clients:
    #         if ids is not None:
    #             # get the intersecting list between ids and motor_ids
    #             ids_to_enable = list(set(open_client.motor_ids) & set(ids))
    #             print(f"\nEnabling motor id {ids_to_enable}\n")
    #             open_client.set_torque_enabled(ids_to_enable, True)
    #         else:
    #             print("\nEnabling all the motors\n")
    #             open_client.set_torque_enabled(open_client.motor_ids, True)

    def set_kp_list(self, kp: List[int]):
        """Set the proportional gain (Kp) for the motors.

        This method updates the proportional gain values for the specified motors by writing to their control table.

        Args:
            kp (List[int]): A list of proportional gain values to be set for the motors.
        """
        assert np.all(np.array(kp) < 0xff) and np.all(0 <= np.array(kp))
        with self.lock:
            # self.client.sync_write(motor_ids=self.motor_ids,
            #                        params=[bytes([_p]) for _p in kp],
            #                        address=SMS_STS_EEPROM_Table_RW.KP,
            #                        size=1)
            self.client.sync_write_1_byte(motor_ids=self.motor_ids,
                                   params=[bytes([_p]) for _p in kp],
                                   address=SMS_STS_EEPROM_Table_RW.KP)


    def set_same_kp_kd(self, kp: int, kd: int):
        """Set same proportional (kp) and derivative (kd) gains for all motor under client.

        This function updates the motor's control parameters by writing the specified
        proportional and derivative gains to the motor's registers. The operation is
        thread-safe.

        Args:
            kp (int): The proportional gain to be set for the motor.
            kd (int): The derivative gain to be set for the motor.
        """
        assert 0 <= kp < 0xff and 0 <= kd < 0xff
        logger.info(f"Setting motor kp={kp} kd={kd}")
        with self.lock:
            self.client.sync_write_2_bytes(motor_ids=self.motor_ids,
                                         params=[bytes([kp, kd])] * len(self.motor_ids),
                                         address=SMS_STS_EEPROM_Table_RW.KP) # kp, kd


    # def set_parameters(self, kp=None, kd=None, ki=None, kff1=None, kff2=None, ids=None):
    #     """Sets the motor control parameters for specified Feite motors.
    #
    #     This function updates the proportional (kp), derivative (kd), integral (ki),
    #     and feedforward (kff1, kff2) gains for the motors identified by the given IDs.
    #
    #     Args:
    #         kp (int, optional): Proportional gain. If None, the parameter is not updated.
    #         kd (int, optional): Derivative gain. If None, the parameter is not updated.
    #         ki (int, optional): Integral gain. If None, the parameter is not updated.
    #         kff1 (int, optional): First feedforward gain. If None, the parameter is not updated.
    #         kff2 (int, optional): Second feedforward gain. If None, the parameter is not updated.
    #         ids (list of int, optional): List of motor IDs to update. If None, no motors are updated.
    #     """
    #     logger.info("Setting motor parameters")
    #     with self.lock:
    #         if kp is not None:
    #             self.client.sync_write(ids, [kp], 84, 2)
    #         if kd is not None:
    #             self.client.sync_write(ids, [kd], 80, 2)
    #         if ki is not None:
    #             self.client.sync_write(ids, [ki], 82, 2)
    #         if kff1 is not None:
    #             self.client.sync_write(ids, [kff1], 90, 2)
    #         if kff2 is not None:
    #             self.client.sync_write(ids, [kff2], 88, 2)

    # @profile()
    def set_pos(self, pos: List[float]):
        """Sets the position of the motors by updating the desired position.

        Args:
            pos (List[float]): A list of position values to set for the motors.
        """
        pos_arr: npt.NDArray[np.float32] = np.array(pos)
        pos_arr_drive = self.init_pos + pos_arr
        with self.lock:
            self.client.write_desired_pos(motor_ids=self.motor_ids,positions=pos_arr_drive)

    # @profile()
    def get_motor_state(self, retries: int = 0) -> Dict[int, JointState]:
        """Retrieves the current state of the motors, including position, velocity, and current.

        Args:
            retries (int): The number of retry attempts for reading motor data in case of failure. Defaults to 0.

        Returns:
            Dict[int, JointState]: A dictionary mapping motor IDs to their respective `JointState`, which includes time, position, velocity, and torque.
        """

        # log(f"Start... {time.time()}", header="Feite", level="warning")
        state_dict: Dict[int, JointState] = {}
        record: PosVelLoadRecord = None
        with self.lock:
            # time, pos_arr = self.client.read_pos(retries=retries)
            # time, pos_arr, vel_arr = self.client.read_pos_vel(retries=retries)

            record = self.client.read_pos_vel_load(
                retries=retries
            )

        # log(f"Pos: {np.round(pos_arr, 4)}", header="Feite", level="debug")
        # log(f"Vel: {np.round(vel_arr, 4)}", header="Feite", level="debug")
        # log(f"Cur: {np.round(cur_arr, 4)}", header="Feite", level="debug")

        # relative to init pos.
        relative_pos = record.pos - self.init_pos

        for _id in self.motor_ids:
            state_dict[_id] = JointState(
                time=record.comm_time,
                pos=relative_pos[i]
                vel=vel_arr[i], tor=load_arr[i]
            )

        # log(f"End... {time.time()}", header="Feite", level="warning")

        return state_dict
