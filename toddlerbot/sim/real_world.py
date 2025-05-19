import platform
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Dict, List, Optional

import numpy as np

from ..actuation import JointState
from ..sim import BaseSim, Obs
from ..sim.robot import Robot
from ..sensing.IMU import IMU as BNO08X_IMU
from ..utils.file_utils import find_ports
from ..actuation import BaseController
from ..actuation.dynamixel_control import (DynamixelConfig, DynamixelController)
from ..actuation.feite_control import (FeiteConfig, FeiteController)

from ._module_logger import logger

def _init_dynamixel_actuators(*, robot:Robot, executor: ThreadPoolExecutor)->Future:
    # from ..actuation.dynamixel_control import (
    #     DynamixelConfig,
    #     DynamixelController,
    # )

    os_type = platform.system()

    description = (
        "USB Serial Port"
        if os_type == "Windows"
        else "USB <-> Serial Converter"
    )

    dynamixel_ports: List[str] = find_ports(description)

    dynamixel_ids = robot.get_joint_attrs("type", "dynamixel", "id")

    dynamixel_ids = robot.motorordering......

    dynamixel_config = DynamixelConfig(
        port=dynamixel_ports[0],
        baudrate=robot.config["general"]["dynamixel_baudrate"],
        control_mode=robot.get_joint_attrs(
            "type", "dynamixel", "control_mode"
        ),
        kP=robot.get_joint_attrs("type", "dynamixel", "kp_real"),
        kI=robot.get_joint_attrs("type", "dynamixel", "ki_real"),
        kD=robot.get_joint_attrs("type", "dynamixel", "kd_real"),
        kFF2=robot.get_joint_attrs("type", "dynamixel", "kff2_real"),
        kFF1=robot.get_joint_attrs("type", "dynamixel", "kff1_real"),
        init_pos=robot.get_joint_attrs("type", "dynamixel", "init_pos"),
    )
    return executor.submit(
        DynamixelController, dynamixel_config, dynamixel_ids
    )

def _init_feite_actuators(*, robot:Robot, executor: ThreadPoolExecutor)-> Optional[Future]:
    # from ..actuation.feite_control import FeiteController
    # TODO: write into YAML.
    
    descriptions = (        
        'USB Serial',  # CH340
        'FT232R USB UART'  # FT232R
    )

    feite_ports: List[str] | None = None
    
    for _dsc in descriptions:        
         if len(found:= find_ports(_dsc)) != 0:
             feite_ports = found
             break
            
    if feite_ports is None:
        raise EnvironmentError(f'can not find any com port connecting feite motors. used descriptions: { descriptions }')            
                    
    feite_ids = robot.get_joint_attrs("type", "feite", "id")

    feite_config = FeiteConfig(
        port=feite_ports[0],
        baudrate=robot.config["general"]["feitei_baudrate"],
        control_mode=robot.get_joint_attrs(
            "type", "feite", "control_mode"
        ),
        kP=robot.get_joint_attrs("type", "feite", "kp_real"),
        kI=robot.get_joint_attrs("type", "feite", "ki_real"),
        kD=robot.get_joint_attrs("type", "feite", "kd_real"),
        # kFF2=robot.get_joint_attrs("type", "dynamixel", "kff2_real"),
        # kFF1=robot.get_joint_attrs("type", "dynamixel", "kff1_real"),
        init_pos=robot.get_joint_attrs("type", "feite", "init_pos"),
    )
    return executor.submit(FeiteController,feite_config, feite_ids)


def  _init_imu(executor: ThreadPoolExecutor)->Future:
    # from toddlerbot.sensing.IMU import IMU
    return executor.submit(BNO08X_IMU)


class RealWorld(BaseSim):
    """Real-world robot interface class."""

    def __init__(self, robot: Robot):
        """Initializes the real-world robot interface.

        Args:
            robot (Robot): An instance of the Robot class containing configuration details.

        Attributes:
            has_imu (bool): Indicates if the robot is equipped with an Inertial Measurement Unit (IMU).
            has_dynamixel (bool): Indicates if the robot uses Dynamixel motors.
            negated_motor_names (List[str]): A list of motor names that require direction negation due to URDF configuration issues.
        """
        super().__init__("real_world")
        self.robot = robot

        # self.imu is not None:bool = self.robot.config["general"]["has_imu"]
        # self.has_dynamixel:bool = self.robot.config["general"]["has_dynamixel"]
        # self.has_feite:bool = self.robot.config["general"]["has_feite"]

        # TODO: Fix the mate directions in the URDF and remove the negated_motor_names
        self.negated_motor_names: List[str] = [
            "neck_pitch_act",
            "left_sho_roll",
            "right_sho_roll",
            "left_elbow_roll",
            "right_elbow_roll",
            "left_wrist_pitch_drive",
            "right_wrist_pitch_drive",
            "left_gripper_rack",
            "right_gripper_rack",
        ]

        self._executor = ThreadPoolExecutor(max_workers=4)  # default max num of threads: (os.cpu_count() or 1) + 4. no need so much occupied.

        # TODO: not expose dynamixel_controller and imu to external modules, such as run_policy. cause this I/O bound call should
        # make use of ThreadPoolExecutor.
        # self.dynamixel_controller: DynamixelController|None = None
        # self.feite_controller: FeiteController | None = None

        self.actuator_controller: BaseController | None = None
        self.imu: BNO08X_IMU | None = None
        
        self.initialize()
        

    def initialize(self) :
        """Initializes the robot's components, including IMU and Dynamixel controllers, if available.

        This method sets up a thread pool _executor to initialize the IMU and Dynamixel controllers asynchronously. It checks the operating system type to determine the appropriate port description for Dynamixel communication. If the robot is configured with an IMU, it initializes the IMU in a separate thread. Similarly, if the robot has Dynamixel actuators, it configures and initializes the Dynamixel controller using the specified port, baud rate, and control parameters. After initialization, it retrieves the results of the asynchronous operations and assigns them to the respective attributes. Finally, it performs a series of observations to ensure the components are functioning correctly.
        """
        future_seq: Dict[Future, str] = {}
        # only allow one type actuator existing.
        assert (self.robot.config["general"]["has_dynamixel"] ^ self.robot.config["general"]["has_feite"])

        if self.robot.config["general"]["has_dynamixel"]:
            dyn_f = _init_dynamixel_actuators(robot=self.robot, executor=self._executor)
            if dyn_f is not None:
                future_seq[dyn_f] = 'actuator_controller'  # attr name.

        if self.robot.config["general"]["has_feite"]:
            ft_f = _init_feite_actuators(robot=self.robot, executor=self._executor)
            if ft_f is not None:
                future_seq[ft_f] = 'actuator_controller'  # attr name.
                
        if self.robot.config["general"]["has_imu"]:
            imu_f = _init_imu(self._executor)
            if imu_f is not None:
                future_seq[imu_f] = 'imu'  # attr name.

        for _f in as_completed(future_seq):
            attr:str = future_seq[_f]
            try:
                value = _f.result()
            except Exception as exc:
                setattr(self, attr, None)
                logger.error(f'instantiate {attr} generated an exception: {exc}')                
            else:
                setattr(self, attr, value)
                logger.info(f'instantiate {attr} succeed.')

        for _ in range(100):
            self.get_observation()

    # @profile()
    def process_motor_reading(self, results: Dict[str, Dict[int, JointState]]) -> Optional[Obs]:
        """Processes motor readings and returns an observation object.

        Args:
            results (Dict[str, Dict[int, JointState]]): A dictionary containing motor state data, indexed by motor type and ID.

        Returns:
            Obs: An observation object containing the current time, motor positions, velocities, and torques.
        """
        motor_state_dict_unordered: Dict[str, JointState] = {}
        if self.actuator_controller is None:
            return None

        actuator_state = results["actuator"]

        for motor_name in self.robot.get_joint_attrs("type", "dynamixel"):
            motor_id = self.robot.config["joints"][motor_name]["id"]
            motor_state_dict_unordered[motor_name] = actuator_state[motor_id]

        time_curr = 0.0
        motor_pos = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_vel = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_tor = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        for i, motor_name in enumerate(self.robot.motor_ordering):
            if i == 0:
                time_curr = motor_state_dict_unordered[motor_name].time

            if motor_name in self.negated_motor_names:
                motor_pos[i] = -motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = -motor_state_dict_unordered[motor_name].vel
            else:
                motor_pos[i] = motor_state_dict_unordered[motor_name].pos
                motor_vel[i] = motor_state_dict_unordered[motor_name].vel

            motor_tor[i] = abs(motor_state_dict_unordered[motor_name].tor)

        obs = Obs(
            time=time_curr,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
        )
        return obs

    def step(self):
        pass

    # @profile()
    def get_observation(self, retries: int = 0):
        """Retrieve and process sensor observations asynchronously.

        This method collects data from available sensors, such as Dynamixel motors and IMU, using asynchronous calls. It processes the collected data to generate a comprehensive observation object.

        Args:
            retries (int, optional): The number of retry attempts for obtaining motor state data. Defaults to 0.

        Returns:
            An observation object containing processed sensor data, including motor states and, if available, IMU angular velocity and Euler angles.
        """
        results: Dict[str, Any] = {}
        futures: Dict[str, Any] = {}
        if self.actuator_controller is not None:
            # results["dynamixel"] = self.actuator_controller.get_motor_state(retries)
            futures["dynamixel"] = self._executor.submit(
                self.actuator_controller.get_motor_state, retries
            )

        if self.imu is not None:
            # results["imu"] = self.imu.get_state()
            futures["imu"] = self._executor.submit(self.imu.get_state)

        # start_times = {key: time.time() for key in futures.keys()}
        for future in as_completed(futures.values()):
            for key, f in futures.items():
                if f is future:
                    # end_time = time.time()
                    results[key] = future.result()
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

        obs = self.process_motor_reading(results)

        if self.imu is not None:
            obs.ang_vel = np.array(results["imu"]["ang_vel"], dtype=np.float32)
            obs.euler = np.array(results["imu"]["euler"], dtype=np.float32)

        return obs

    # @profile()
    def set_motor_target(self, motor_angles: Dict[str, float]):
        """Sets the target angles for the robot's motors, adjusting for any negated motor directions and updating the positions of Dynamixel motors if present.

        Args:
            motor_angles (Dict[str, float]): A dictionary mapping motor names to their target angles in degrees.
        """

        # Directions are tuned to match the assembly of the robot.
        motor_angles_updated: Dict[str, float] = {}
        for name, angle in motor_angles.items():
            if name in self.negated_motor_names:
                motor_angles_updated[name] = -angle
            else:
                motor_angles_updated[name] = angle

        if self.actuator_controller is not None:
            dynamixel_pos = [
                motor_angles_updated[k]
                for k in self.robot.get_joint_attrs("type", "dynamixel")
            ]
            self._executor.submit(self.actuator_controller.set_pos, dynamixel_pos)

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for motors of type 'dynamixel'.

        If the robot has Dynamixel motors, this method updates their Kp values based on the provided dictionary. If a motor's Kp is not specified in the dictionary, it defaults to the value in the robot's configuration.

        Args:
            motor_kps (Dict[str, float]): A dictionary mapping motor names to their desired Kp values.
        """

        if self.actuator_controller is not None:
            dynamixel_kps: List[float] = []
            for k in self.robot.get_joint_attrs("type", "dynamixel"):
                if k in motor_kps:
                    dynamixel_kps.append(motor_kps[k])
                else:
                    dynamixel_kps.append(self.robot.config["joints"][k]["kp_real"])

            self._executor.submit(self.actuator_controller.set_kp, dynamixel_kps)

    def close(self):
        """Closes all active components and shuts down the _executor.

        This method checks for active components such as Dynamixel motors and IMU sensors.
         If they are present, it submits tasks to close them using the _executor.
         Finally, it shuts down the _executor, ensuring all submitted tasks are completed before termination.
        """

        if self.actuator_controller is not None:
            self._executor.submit(self.actuator_controller.close_motors)

        if self.imu is not None:
            self._executor.submit(self.imu.close)

        self._executor.shutdown(wait=True)
