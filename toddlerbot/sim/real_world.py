import platform
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Optional, Mapping, Set, Any
from collections import OrderedDict

import numpy as np
import numpy.typing as npt

from ..sensing import IMU as BNO08X_IMU
from ..utils import find_ports
from ..actuation import *

from .base_env import BaseEnv, Obs
from .robot import Robot
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
    # dynamixel_ids = robot.get_joint_config_attrs("type", "dynamixel", "id")

    # NOTE: must use the ordered id list as input to controller.
    dynamixel_ids = robot.motor_id_ordering

    control_mode = robot.get_motor_ordered_config_attrs(
        "type", "dynamixel", "control_mode"
    )
    assert len(control_mode) == len(dynamixel_ids)

    kP = robot.get_motor_ordered_config_attrs("type", "dynamixel", "kp_real")
    assert len(kP) == len(dynamixel_ids)

    kI = robot.get_motor_ordered_config_attrs("type", "dynamixel", "ki_real")
    assert len(kI) == len(dynamixel_ids)

    kD = robot.get_motor_ordered_config_attrs("type", "dynamixel", "kd_real")
    assert len(kD) == len(dynamixel_ids)

    kFF2 = robot.get_motor_ordered_config_attrs("type", "dynamixel", "kff2_real")
    assert len(kFF2) == len(dynamixel_ids)

    kFF1 = robot.get_motor_ordered_config_attrs("type", "dynamixel", "kff1_real")
    assert len(kFF1) == len(dynamixel_ids)

    # TODO: why not use robot.default_motor_angles directly.
    init_pos = robot.get_motor_ordered_config_attrs("type", "dynamixel", "init_pos")
    assert len(init_pos) == len(dynamixel_ids)

    dynamixel_config = DynamixelConfig(
        port=dynamixel_ports[0],
        baudrate=robot.config["general"]["dynamixel_baudrate"],
        control_mode=control_mode,
        kP=kP,
        kI=kI,
        kD=kD,
        kFF2=kFF2,
        kFF1=kFF1,
        init_pos=init_pos,
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

    # we use motor_name_ordering directly. no need to get ids through iterating over the json config `type` , causing we don't use two
    # type of motors at same time.
    # feite_ids = robot.get_joint_attrs("type", "feite", "id")

    # NOTE: must use the ordered id list as input to controller.
    feite_ids = robot.motor_id_ordering

    # control_mode = robot.get_motor_ordered_config_attrs("type", "feite", "control_mode")
    control_mode = np.asarray((robot.motor_control_mode[_n] for _n in robot.motor_name_ordering), dtype=np.float32)
    assert len(control_mode) == len(feite_ids)

    kP = np.asarray((robot.motor_kp_real[_n] for _n in robot.motor_name_ordering), dtype=np.float32)
    # kP = robot.get_motor_ordered_config_attrs("type", "feite", "kp_real")
    assert len(kP) == len(feite_ids)

    # kI = robot.get_motor_ordered_config_attrs("type", "feite", "ki_real")
    kI = np.asarray((robot.motor_ki_real[_n] for _n in robot.motor_name_ordering), dtype=np.float32)
    assert len(kI) == len(feite_ids)

    # kD = robot.get_motor_ordered_config_attrs("type", "feite", "kd_real")
    kD = np.asarray((robot.motor_kd_real[_n] for _n in robot.motor_name_ordering), dtype=np.float32)
    assert len(kD) == len(feite_ids)

    # NOTE: read `init_pos` is written by calibrate_zero.
    # init_pos = robot.get_motor_ordered_config_attrs("type", "feite", "init_pos")
    init_pos = np.asarray((robot.motor_init_pos[_n] for _n in robot.motor_name_ordering), dtype=np.float32)
    assert len(init_pos) == len(feite_ids)

    feite_config = FeiteConfig(
        port=feite_ports[0],
        baudrate=robot.config["general"]["feitei_baudrate"],
        control_mode=control_mode,
        kP=kP,
        kI=kI,
        kD=kD,
        init_pos=init_pos,
    )
    return executor.submit(FeiteController,feite_config, feite_ids)


def  _init_imu(executor: ThreadPoolExecutor)->Future:
    # from toddlerbot.sensing.IMU import IMU
    return executor.submit(BNO08X_IMU)

class RealWorld(BaseEnv, env_name='real_world'):
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
        # super().__init__("real_world")
        super().__init__()
        self.robot = robot

        # self._imu is not None:bool = self.robot.config["general"]["has_imu"]
        # self.has_dynamixel:bool = self.robot.config["general"]["has_dynamixel"]
        # self.has_feite:bool = self.robot.config["general"]["has_feite"]

        # # TODO: Fix the mate directions in the URDF and remove the negated_motor_names
        # self.negated_motor_names: List[str] = [
        #     "neck_pitch_act",
        #     "left_sho_roll",
        #     "right_sho_roll",
        #     "left_elbow_roll",
        #     "right_elbow_roll",
        #     "left_wrist_pitch_drive",
        #     "right_wrist_pitch_drive",
        #     "left_gripper_rack",
        #     "right_gripper_rack",
        # ]

        self._executor = ThreadPoolExecutor(max_workers=5)  # default max num of threads: (os.cpu_count() or 1) + 4. no need so much occupied.

        # TODO: not expose dynamixel_controller and _imu to external modules, such as run_policy. cause this I/O bound call should
        # make use of ThreadPoolExecutor.
        # self.dynamixel_controller: DynamixelController|None = None
        # self.feite_controller: FeiteController | None = None

        # TODO: should put actuator_controller, _imu into Robot.
        self.actuator_controller: BaseController | None = None
        self._imu: BNO08X_IMU | None = None
        # TODO: only RealWorld has negated_motor_list, MujocoSim has no .
        # self.negated_motor_ids: List[int] = []
        self.negated_motor_direction_mask: npt.NDArray[np.float32] | None = None
        
        self.initialize()

    def initialize(self) :
        """Initializes the robot's components, including IMU and Dynamixel controllers, if available.

        This method sets up a thread pool _executor to initialize the IMU and Dynamixel controllers asynchronously. It checks the operating system type to determine the appropriate port description for Dynamixel communication. If the robot is configured with an IMU, it initializes the IMU in a separate thread. Similarly, if the robot has Dynamixel actuators, it configures and initializes the Dynamixel controller using the specified port, baud rate, and control parameters. After initialization, it retrieves the results of the asynchronous operations and assigns them to the respective attributes. Finally, it performs a series of observations to ensure the components are functioning correctly.
        """
        future_seq: Dict[Future, str] = {}
        general_cfg: Dict[str, Any] = self.robot.config["general"]
        # only allow one type actuator existing.
        assert ( ('has_dynamixel' in general_cfg and general_cfg['has_dynamixel']) ^
                 ( "has_feite" in general_cfg and general_cfg["has_feite"] ))

        if 'has_dynamixel' in general_cfg and general_cfg["has_dynamixel"]:
            dyn_f = _init_dynamixel_actuators(robot=self.robot, executor=self._executor)
            if dyn_f is not None:
                future_seq[dyn_f] = 'actuator_controller'  # attr name.

        if 'has_feite' in general_cfg and general_cfg["has_feite"]:
            ft_f = _init_feite_actuators(robot=self.robot, executor=self._executor)
            if ft_f is not None:
                future_seq[ft_f] = 'actuator_controller'  # attr name.
                
        if general_cfg["has_imu"]:
            imu_f = _init_imu(self._executor)
            if imu_f is not None:
                future_seq[imu_f] = '_imu'  # attr name.

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

        # TODO: Fix the mate directions in the URDF and remove the negated_motor_names
        _neg_mtr_names: Set[str] = {
            "neck_pitch_act",
            "left_sho_roll",
            "right_sho_roll",
            "left_elbow_roll",
            "right_elbow_roll",
            "left_wrist_pitch_drive",
            "right_wrist_pitch_drive",
            "left_gripper_rack",
            "right_gripper_rack",
        }

        # _neg_mtr_names: List[str] = [
        #     "neck_pitch_act",
        #     "left_sho_roll",
        #     "right_sho_roll",
        #     "left_elbow_roll",
        #     "right_elbow_roll",
        #     "left_wrist_pitch_drive",
        #     "right_wrist_pitch_drive",
        #     "left_gripper_rack",
        #     "right_gripper_rack",
        # ]

        # _negated_motor_ids: List[int] = [
        #     self.robot.motor_name_to_id[_name] for _name in _neg_mtr_names
        # ]

        self.negated_motor_direction_mask: npt.NDArray[np.float32] = np.where(
            # np.isin(self.robot.motor_id_ordering, _negated_motor_ids),
            np.isin(self.robot.motor_name_ordering, _neg_mtr_names),
            -1.,
            1. ).astype(np.float32)

        assert len(self.negated_motor_direction_mask) == self.robot.nu

        for _ in range(100):
            self.get_observation(1)

    def post_process_motor_reading(self, motor_state: Mapping[int, JointState])\
            -> Dict[str, float|npt.NDArray[np.float32]]:
        """Processes motor readings and returns an observation object.

        Args:
            # results (Dict[str, Dict[int, JointState]]): A dictionary containing motor state data, indexed by motor type and ID.
            motor_state:  key is motor id.

        Returns:
            Obs: An observation object containing the current time, motor positions, velocities, and torques.
        """
        # motor_state_dict_unordered: Dict[str, JointState] = {}

        assert len(motor_state) == self.robot.nu

        # for motor_name in self.robot.get_joint_config_attrs("type", "dynamixel"):
        #     motor_id = self.robot.config["joints"][motor_name]["id"]
        #     motor_state_dict_unordered[motor_name] = actuator_state[motor_id]


        # time_curr = 0.0
        motor_pos = np.zeros(self.robot.nu,dtype=np.float32)    #  len(self.robot.motor_name_ordering), dtype=np.float32)
        # motor_vel = np.zeros(self.robot.nu,dtype=np.float32)    #   len(self.robot.motor_name_ordering), dtype=np.float32)
        # motor_tor = np.zeros(self.robot.nu,dtype=np.float32)    #len(self.robot.motor_name_ordering), dtype=np.float32)
        motor_vel = np.zeros_like(motor_pos)
        motor_tor = np.zeros_like(motor_pos)

        time_curr = motor_state[self.robot.motor_id_ordering[0]].time
        for _x, _id in enumerate(self.robot.motor_id_ordering):
            # if _id in self.negated_motor_ids:
            #     motor_pos[_x] = - motor_state[_id].pos
            #     motor_vel[_x] = - motor_state[_id].vel
            # else:
            #     motor_pos[_x] = motor_state[_id].pos
            #     motor_vel[_x] = motor_state[_id].vel

            motor_pos[_x] = motor_state[_id].pos
            motor_vel[_x] = motor_state[_id].vel
            motor_tor[_x] = abs(motor_state[_id].tor)

        motor_pos *= self.negated_motor_direction_mask
        motor_vel *= self.negated_motor_direction_mask

        # for i, motor_name in enumerate(self.robot.motor_name_ordering):
        #     if i == 0:
        #         time_curr = motor_state_dict_unordered[motor_name].time
        #
        #     if motor_name in self.negated_motor_names:
        #         motor_pos[i] = -motor_state_dict_unordered[motor_name].pos
        #         motor_vel[i] = -motor_state_dict_unordered[motor_name].vel
        #     else:
        #         motor_pos[i] = motor_state_dict_unordered[motor_name].pos
        #         motor_vel[i] = motor_state_dict_unordered[motor_name].vel
        #
        #     motor_tor[i] = abs(motor_state_dict_unordered[motor_name].tor)

        # obs = Obs(
        #     time=time_curr,
        #     motor_pos=motor_pos,
        #     motor_vel=motor_vel,
        #     motor_tor=motor_tor,
        # )
        # return obs

        return dict(
            time=time_curr,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,)

    def step(self):
        pass

    def read_motor_state(self, retries:int) ->Optional[Mapping[str, float|npt.NDArray[np.float32]]]:
        if self.actuator_controller is not None:
            ste: Mapping[int, JointState] = self.actuator_controller.get_motor_state(retries)
            # for _k,_v in self.process_motor_reading(state):
            #       setattr(obs, __name=_k, __value=_v)
            # if m_ste is None:
            #     obs.time = np.inf
            #     obs.motor_pos = np.full(shape=self.robot.nu,fill_value=np.inf, dtype=np.float32)
            #     obs.motor_vel = obs.motor_pos.copy()
            #     obs.motor_tor = obs.motor_pos.copy()
            # else:
            #     processed = self.post_process_motor_reading(ste)
            # TODO: copy to new array?
            # obs.time = processed['time']
            # obs.motor_pos = processed['motor_pos']
            # obs.motor_vel = processed['motor_vel']
            # obs.motor_tor = processed['motor_tor']
            return self.post_process_motor_reading(ste)
        else:
            return None

    def read_imu_state(self, retries: int) ->Optional[Mapping[str, float|npt.NDArray[np.float32]]]:
        if self._imu is not None:
            #         if self._imu is not None:
            #             def _set_obs_imu(i_ste: Mapping[str, npt.NDArray[np.float32]]):
            #                 if i_ste is None:
            #                     obs.euler.fill(np.inf)
            #                     obs.ang_vel.fill(np.inf)
            #                 else:
            #                     copy to new array?
            #
            #                     obs.euler = i_ste['euler']
            #                     obs.ang_vel = i_ste['ang_vel']

            return self._imu.get_state()
        else:
            return None


    # @profile()
    def get_observation(self, retries:int = 0) ->Optional[Obs]:
        """Retrieve and process sensor observations asynchronously.

        This method collects data from available sensors, such as Dynamixel motors and IMU, using asynchronous calls. It processes the collected data to generate a comprehensive observation object.

        Args:
            retries (int, optional): The number of retry attempts for obtaining motor state data. Defaults to 0.

        Returns:
            An observation object containing processed sensor data, including motor states and, if available, IMU angular velocity and Euler angles.
        """
        # results: Dict[str, Any] = {}
        # futures: Dict[str, Any] = {}
        # future_seq: Dict[Future, Callable[[Mapping[int|str,Any] | None],None]] = {}
        obs = Obs()
        future_seq: Dict[Future, str] = {
            self._executor.submit(self.read_motor_state, retries): 'read_motor_state',
            self._executor.submit(self.read_imu_state, retries): 'read_imu_state',
        }

        # results["dynamixel"] = self.actuator_controller.get_motor_state(retries)
        # results["_imu"] = self._imu.get_state()

        for _f in as_completed(future_seq):
            read_sensor = future_seq[_f]
            try:
                ste: Mapping[str, float|npt.NDArray[np.float32]] = _f.result()
            except Exception as exc:
                # let the corresponding attrs in obs be inited `None`.
                logger.error(f' {read_sensor} generated an exception: {exc}')
            else:
                for _k,_v in ste:
                    if hasattr(obs, _k):
                        setattr(obs, __name=_k, __value=_v)
                    else:
                        raise ValueError(f'read state key: {_k} not in obs.')

                logger.debug(f' {read_sensor} succeed, got obs keys: {ste.keys()} ')

        # # start_times = {key: time.time() for key in futures.keys()}
        # for future in as_completed(futures.values()):
        #     for key, f in futures.items():
        #         if f is future:
        #             # end_time = time.time()
        #             results[key] = future.result()
        #             # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
        #             break
        #
        # obs = self.process_motor_reading(results)
        #
        # if self._imu is not None:
        #     obs.ang_vel = np.array(results["_imu"]["ang_vel"], dtype=np.float32)
        #     obs.euler = np.array(results["_imu"]["euler"], dtype=np.float32)

        return obs

    # @profile()
    def set_motor_target(self, motor_angles: Dict[str, float]| npt.NDArray[np.float32]):
        """Sets the target angles for the robot's motors, adjusting for any negated motor directions and updating
         the positions of Dynamixel motors if present.

        Args:
            motor_angles (Dict[str, float]): A dictionary mapping motor names to their target angles in degrees
             or a NumPy array of target angles. If a dictionary is provided, the values are converted to a NumPy array of type float32.
        """

        # Directions are tuned to match the assembly of the robot.
        # motor_angles must contain all the robot motors.
        assert len(motor_angles)==self.robot.nu
        if self.actuator_controller is None:
            raise ValueError(f'self.actuator_controller is None.')

        write_pos: npt.NDArray[np.float32]= np.full_like(self.robot.motor_name_ordering,
                                                         fill_value=np.inf,
                                                         dtype=np.float32)

        if isinstance(motor_angles, (dict, OrderedDict)):
            for _x, _n in enumerate(self.robot.motor_name_ordering):
                # assert _name in motor_angles
                write_pos[_x]=(motor_angles[_n])

        elif isinstance(motor_angles, npt.NDArray[np.float32]):
            write_pos = motor_angles

        else:
            raise TypeError(f'motor_angles type error: {type(motor_angles)=:} ')

        assert np.all(write_pos != np.inf)
        # write_pos *= self.negated_motor_direction_mask

        # TODO: not waiting for the future to complete?
        self._executor.submit(self.actuator_controller.set_pos,
                              write_pos * self.negated_motor_direction_mask)

        # for _id, _name in zip(self.robot.motor_id_ordering, self.robot.motor_name_ordering):
        # if isinstance(motor_angles, (dict, OrderedDict)):
        #     for _id in self.robot.motor_id_ordering:
        #         _name = self.robot.id_to_motor_name[_id]
        #         # assert _name in motor_angles
        #
        #         if _id in self.negated_motor_ids:
        #             write_pos.append( -motor_angles[_name])
        #         else:
        #             write_pos.append(motor_angles[_name])
        # elif isinstance(motor_angles, npt.NDArray[np.float32]):
        #     write_pos = list(motor_angles)
        #     for _id in self.robot.motor_id_ordering:
        #         if _id in self.negated_motor_ids:
        #             write_pos.append(-motor_angles[_name])
        #         else:
        #             write_pos.append(motor_angles[_name])
        #
        # else:
        #     raise TypeError(f'motor_angles type error: {type(motor_angles)=:} ')

        # assert len(write_pos) == len(motor_angles)

        # motor_angles_updated: Dict[int, float] = {}
        # for _name, _angle in motor_angles.items():
        #     mtr_id = self.robot.motor_name_to_id[_name]
        #
        #     if mtr_id in self.negated_motor_ids:
        #         motor_angles_updated[mtr_id] = - _angle
        #     else:
        #         motor_angles_updated[mtr_id] = _angle

        # if self.actuator_controller is not None:
        #     dynamixel_pos = [
        #         motor_angles_updated[k]
        #         for k in self.robot.get_joint_config_attrs("type", "dynamixel")
        #     ]

        # # TODO: not waiting for the future to complete?
        # self._executor.submit(self.actuator_controller.set_pos, write_pos)

    # NOTE: sync write to all.
    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for motors of type 'dynamixel'.

        If the robot has Dynamixel motors, this method updates their Kp values based on the provided dictionary.
        If a motor's Kp is not specified in the dictionary, it defaults to the value in the robot's configuration.

        Args:
            motor_kps (Dict[str, float]): A dictionary mapping motor names to their desired Kp values.
        """
        # assert len(motor_kps) == self.robot.nu

        if self.actuator_controller is None:
            raise ValueError(f'self.actuator_controller is None.')

        write_kps: List[float] = []

        # for k in self.robot.get_joint_config_attrs("type", "dynamixel"):
        for _name in self.robot.motor_name_ordering:
            if _name in motor_kps:
                write_kps.append(motor_kps[_name])
            else:
                write_kps.append(self.robot.config["joints"][_name]["kp_real"])

        # TODO: not waiting for the future to complete?
        self._executor.submit(self.actuator_controller.set_kp, write_kps)

    # TODO: use context manager.
    def close(self):
        """Closes all active components and shuts down the _executor.

        This method checks for active components such as Dynamixel motors and IMU sensors.
         If they are present, it submits tasks to close them using the _executor.
         Finally, it shuts down the _executor, ensuring all submitted tasks are completed before termination.
        """

        if self.actuator_controller is not None:
            self._executor.submit(self.actuator_controller.close_motors)

        if self._imu is not None:
            self._executor.submit(self._imu.close)

        self._executor.shutdown(wait=True)
