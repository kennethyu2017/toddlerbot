"""
experimented for robstride RS02 actuator.  by kenneth yu.
"""

import time
from typing import Dict, NamedTuple, Sequence, Tuple
import numpy as np
import numpy.typing as npt
import asyncio
# from aioconsole import aprint

from toddlerbot.actuation.robstride_client import RobStrideClient, RSBaudRate,RSRunMode
from toddlerbot.actuation._module_logger import logger
from toddlerbot.actuation.base_controller import BaseController,JointState

class RobStrideConfig(NamedTuple):
    channel: str
    baud_rate: RSBaudRate
    control_mode: RSRunMode
    motor_can_id: Sequence[int]
    host_can_id: int
    pos_kp: Sequence[float] | None = None
    # vel_kp: Sequence[int] | None
    # vel_ki: Sequence[int] | None
    # TODO: adjust according to tracking result.
    # TODO: move into config.json.
    # default_torque_limit: npt.NDArray[np.uint16] |None = None

    default_accel_PP_mode: npt.NDArray[np.float32] |None = None   # [1.6 * np.pi]
    default_vel_PP_mode: npt.NDArray[np.float32] |None = None      # [1.4 * np.pi]
    init_target_pos: npt.NDArray[np.float32] |None = None    # None = None

    # interp_method: str = "cubic"

class RobStrideController(BaseController):
    """Class for controlling RobStride RS02 motors."""

    def __init__(self, config: RobStrideConfig ):
        """Initializes the motor controller with the given configuration and motor IDs.

        Args:
            config
            # motor_ids

        Attributes:
            config (DynamixelConfig): Stores the configuration settings.
            # motor_ids (List[int]): Stores the list of motor IDs.
            # lock (Lock): A threading lock to ensure thread-safe operations.
            # init_pos (np.ndarray): An array of initial positions for the motors, initialized to zeros if not provided in the config.
        """
        client: RobStrideClient
        _motor_ids: Tuple[int]

        logger.info(f'init robstride controller with target motor ids: {config.motor_can_id}'
                    f'\n with config: {config} ')

        self.config = config
        # NOTE: the index in self._motor_ids is used for read data array index, like pos,vel,etc.
        # we use immutable tuple instead of set/list.
        self._motor_id = tuple(set(config.motor_can_id))
        if len(self._motor_id) != len(config.motor_can_id):
            raise ValueError(f'input config motor_can_id include duplicated values: {config.motor_can_id=:}')

        # self.lock = Lock()

        self.client = RobStrideClient(motor_can_id=self._motor_id,
                                      host_can_id=config.host_can_id,
                                      channel=config.channel,
                                      baud_rate=config.baud_rate,
                                      )

        # self.initialize_motors()

        # # NOTE: first set goal pos to init_pos in config.json, then normalize init pos read from motor.
        # # if config.init_pos is None, that is for calibrate_zero.
        # # TODO: during calibrate_zero , setting init_pos to pi ??
        # self.normalized_init_pos: npt.NDArray[np.float32] | None = None
        #
        # if self.config.init_targe_pos is None:
        #     self.normalized_init_pos = np.zeros(len(self._motor_id), dtype=np.float32)
        # else:
        #     assert len(config.init_goal_pos) == len(self._motor_id)
        #     # self.normalized_init_pos = np.asarray(config.init_pos, dtype=np.float32)
        #
        #     self.normalize_init_pos()

    # used for asyncio.
    async def send_rcv_task(self):
        await self.client.send_rcv_task()

    # called after send_rcv_task running in loop.
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
        time.sleep(0.2)

        return

        _, v_in = self.client.read_vin()
        assert len(v_in)==len(self._motor_id)
        logger.info(f"Voltage of motors: (V): {v_in}")
        if np.any(v_in < 10):
            raise ValueError(
                "Voltage too low. Please check the power supply or charge the batteries."
            )

        time.sleep(0.2)

        # ---- TODO: add overload protect, min/max pos.. to Feite motors. ----

        self.client.set_return_delay_time(self.config.return_delay_us)

        self.client.set_control_mode(value= self.config.control_mode )

        # write kP,kD,kI together
        assert np.all(np.array(self.config.kP) <= 0xff) and np.all(0 < np.array(self.config.kP))
        assert np.all(np.array(self.config.kD) <= 0xff) and np.all(0 <= np.array(self.config.kD))
        assert np.all(np.array(self.config.kI) <= 0xff) and np.all(0 <= np.array(self.config.kI))

        self.client.set_kp_kd_ki(kp=self.config.kP, kd=self.config.kD, ki=self.config.kI)

        # check protection:
        _, protect_mode = self.client.read_protect_mode()
        np.set_printoptions(formatter={'int': '0x{:02x}'.format})
        logger.info(f'===> motor protection mode: {protect_mode}')

        if np.any(protect_mode != 0x2c):
            raise ValueError(f'motor protection mode read value:{protect_mode},'
                             f' but every feite motor should be set to 0x2c to enable: overload / over current/ over therm.'
                             f'pls set it and other relative memory table values. ')

        # TODO:
        # check torque limit: EEPROM-16 and SRAM-48
        # check overload torque threshold/protection-duration/protection-torque:  EEPROM-34/35/36

        # TODO: no feedforward of Feite actuator.
        # self.client.sync_write(self._motor_ids, self.config.kFF2, 88, 2)
        # self.client.sync_write(self._motor_ids, self.config.kFF1, 90, 2)
        # self.client.sync_write(self._motor_ids, self.config.current_limit, 102, 2)

        # set acc, vel, adjust present pos as init_pos from config.
        self.client.set_goal_accel(motor_ids=self._motor_id, accel=self.config.default_accel)
        self.client.set_goal_vel(motor_ids=self._motor_id, vel=self.config.default_vel)

        # set torque limit. for safety or perf.
        # TODO: temply set to 90% for sysID.
        self.client.set_torque_limit(motor_ids=self._motor_id, limit_percentage=self.config.default_torque_limit)

        # NOTE: first set goal pos to init_pos in config.json, then normalize init pos read from motor.
        self.client.set_goal_pos(motor_ids=self._motor_id, pos=self.config.init_goal_pos)

        self.client.set_torque_enabled(motor_ids=self._motor_id, enabled=True)

        # NOTE: TO adjust the init pos bias: first set goal pos to init_pos in config.json,
        # then normalize init pos read from motor.
        # if config.init_pos is None, that is for calibrate_zero.
        # TODO: during calibrate_zero , setting init_pos to pi ??
        self.normalized_init_pos: npt.NDArray[np.float32] | None = None

        if self.config.init_target_pos is None:
            # for calibrate_zero.
            self.normalized_init_pos = np.zeros(len(self._motor_can_id), dtype=np.float32)
        else:
            assert len(self._init_target_pos) == len(self._motor_can_id)
            # self.normalized_init_pos = np.asarray(config.init_pos, dtype=np.float32)
            self.normalize_init_pos()

        time.sleep(1.0)

    def _normalize_init_pos(self):
        """Update the initial position to account for any changes in position.

        This method reads the current position from the client and calculates the
        difference from the stored initial position. It then adjusts the initial
        position to reflect any changes, ensuring that the position remains within
        the range of [-π, π].
        """
        _, read_pos = self.client.read_pos(retries=-1)
        # delta_pos = read_pos - self.normalized_init_pos

        delta_pos = read_pos - np.asarray(self.config.init_goal_pos, dtype=np.float32)

        delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi

        self.normalized_init_pos = read_pos - delta_pos

        assert np.all( abs(self.normalized_init_pos) <= np.pi )

        logger.warning(f'====== normalized init pos: {self.normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self.normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self.normalized_init_pos} =============')


    def close_motors(self):
        """Closes all active motor clients.

        This method iterates over all currently open Feite clients and forces them to close if they are in use. It logs a message for each client that is being forcibly closed and then sets the client's port handler to not in use before disconnecting the client.
        """
        open_clients: Set[FeiteGroupClient] = RobStrideClient.OPEN_CLIENTS  # type: ignore
        for _client in open_clients:
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
                # get the intersecting list between ids and _motor_ids
                set_ids = set(_client._motor_can_id) & set(ids)
                logger.info(f"set motor id: {set_ids} toque {enabled=:}")
            else:
                set_ids = _client._motor_can_id
                logger.info(f"set all the motors in client with ids: {set_ids} toque {enabled=:}")

            _client.set_torque_enabled(motor_ids=set_ids, enabled=enabled, retries=2)


    @staticmethod
    def enable_motors(ids:Sequence[int]=None):
        """Enables torque for specified motors or all motors if no IDs are provided.

        Args:
            ids (list, optional): A list of motor IDs to enable. If None, all motors will be enabled.
        """
        # set_motor_torque(enabled=True, ids=ids)

    @staticmethod
    def disable_motors(ids:Sequence[int]=None):
        """Enables torque for specified motors or all motors if no IDs are provided.

        Args:
            ids (list, optional): A list of motor IDs to enable. If None, all motors will be enabled.
        """
        # set_motor_torque(enabled=False, ids=ids)


    def set_kp(self, kp: Sequence[int]):
        """Set the proportional gain (Kp) for the motors.

        This method updates the proportional gain values for the specified motors by writing to their control table.

        Args:
            kp (List[int]): A list of proportional gain values to be set for the motors.
        """
        assert np.all(np.array(kp) <= 0xff) and np.all(0 < np.array(kp))
        self.client.set_kp(kp)


    # NOTE: will offset using self.normalized_init_pos
    def set_pos(self, pos: Sequence[float]):
        """Sets the position of the motors by updating the desired position.

        Args:
            pos (Sequence): A list of position values to set for the motors.
        """

        # TODO: convert from [-pi/2, pi/2] to [pi/2, 3pi/2] -> during calibrate_zero , setting init_pos to pi...

        pos_arr: npt.NDArray[np.float32] = np.array(pos)
        # add init_pos as offset.
        pos_arr_drive = self.normalized_init_pos + pos_arr

        with self.lock:
            self.client.set_goal_pos(motor_ids=self._motor_id, pos=pos_arr_drive)

        # try:
        #     with self.lock:
        #         self.client.set_desired_pos(motor_ids=self._motor_ids, positions=pos_arr_drive)
        # except Exception as err:
        #     logger.error(f' set pos exception: {err} {type(err)}')
        #     raise

    # NOTE: will offset using self.normalized_init_pos
    # @profile()
    def get_motor_state(self, retries: int = 0) -> Dict[int, JointState]:
        """Retrieves the current state of the motors, including position, velocity, and current.

        Args:
            retries (int): The number of retry attempts for reading motor data in case of failure. Defaults to 0.

        Returns:
            Dict[int, JointState]: A dictionary mapping motor IDs to their respective `JointState`, which includes time, position, velocity, and torque.
        """

        # TODO: convert POS from [pi/2, 3pi/2] to [-pi/2, pi/2].

        state_dict: Dict[int, JointState] = {}

        read_value = self.client.read_pos_vel_load(retries=retries)


        assert len(self._motor_id) == len(read_value.pos) == len(read_value.vel) == len(read_value.load)

        # relative to init pos.
        relative_pos = read_value.pos - self.normalized_init_pos

        for _id, _pos, _vel, _load in zip(self._motor_id,
                                          relative_pos,
                                          read_value.vel,
                                          read_value.load) :
            state_dict[_id] = JointState(
                time=read_value.comm_time,
                pos= _pos,
                vel= _vel,
                tor= _load)

        # log(f"End... {time.time()}", header="Feite", level="warning")

        return state_dict

    def connect_to_client(self, usb_com_latency_timer_ms:int, timeout_ms: int):
        raise NotImplementedError


if __name__ == '__main__':
    # import concurrent.futures
    import multiprocessing as mp
    import atexit

    def mock_cpu_bound_policy(ctrl: BaseController):
        time.sleep(2.0)
        print(f'---> start to initialize motors')
        ctrl.initialize_motors()
        print(f'finish initialize motors <---')

        ret : int = 0
        # while True:
        for _ in range(10):
            time.sleep(1.)
            ret+=1
            print(f'---cpu bound --- {ret=:} ----')
            if ret > 0xffffff:
                ret = 0

    def run_io_bound_task_in_spawned_process(ctrl:BaseController):
        return asyncio.run(ctrl.send_rcv_task())

    cfg = RobStrideConfig(channel='can0',
                          baud_rate=RSBaudRate.BPS_1M,
                          control_mode=RSRunMode.PP_POSITION_MODE,
                          host_can_id=0xfe,
                          motor_can_id=[0x7f],
                          pos_kp=None,
                          default_accel_PP_mode=None,
                          default_vel_PP_mode=None,
                          init_target_pos=None)
    controller = RobStrideController(cfg)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as p_pool:
    #     fut:concurrent.futures.Future = p_pool.submit(run_io_bound_task_in_process_pool, controller)

    # NOTE: python `daemon` process mimic the behavour of thread, which will be terminated after the parent process
    # terminates, not the concept of Linux/Unix daemon services which will kept at backgroud even after the parent
    # process terminates.
    io_proc = mp.Process(target=run_io_bound_task_in_spawned_process,
                         args=[controller],
                         name='asyncio_send_rcv_can_msg',
                         daemon=True)


    # terminate gracefully.
    def clean_io_process():
        print(f'clean_io_process(): terminate io_process gracefully.')
        while io_proc.is_alive():
            print(f'io_proc is still alive: {io_proc.is_alive()}, terminate it')
            io_proc.terminate()
            time.sleep(0.5)
        # fut.cancel()
        print(f'io_proc is alive: {io_proc.is_alive()}')
        print(f'close motors:')
        controller.close_motors()
        time.sleep(0.5)

    def exit_handler():
        print(f'called from exit_handler --->')
        clean_io_process()

    atexit.register(exit_handler)

    try:
        print(f'start io_process.')
        io_proc.start()
        # io_proc.join()
        print(f'io_proc is alive: {io_proc.is_alive()}')
        print(f'start mock cpu bound policy.')
        mock_cpu_bound_policy(controller)

    except Exception as exc:
        print(f'--- exception in main process: {exc=:} {type(exc)=:}')
        time.sleep(0.5)
        raise exc

    finally:
        # normal finish.
        print(f'finally: clean up in finally--->')
        time.sleep(0.5)
        clean_io_process()

    # async def run_cpu_bound_policy_in_process_pool(loop: asyncio.AbstractEventLoop, ctrl:BaseController):
    #     try:
    #         with concurrent.futures.ProcessPoolExecutor() as p_pool:
    #             return await loop.run_in_executor(p_pool, mock_cpu_bound_policy, ctrl)
    #
    #     except Exception as exc:
    #         print(f'run_cpu_bound_policy_in_process_pool failed: {exc=:} {type(exc)=:}')
    #         raise exc
    #
    #     finally:
    #         print(f'exit run_cpu_bound_policy_in_process_pool...')

    # async def run_io_bound_task_in_process_pool(loop: asyncio.AbstractEventLoop, ctrl:BaseController):
    #     try:
    #         with concurrent.futures.ProcessPoolExecutor() as p_pool:
    #             return await loop.run_in_executor(p_pool, ctrl.send_rcv_task)
    #
    #     except Exception as exc:
    #         print(f'run_cpu_bound_policy_in_process_pool failed: {exc=:} {type(exc)=:}')
    #         raise exc
    #
    #     finally:
    #         print(f'exit run_cpu_bound_policy_in_process_pool...')

    # async def _main():
    #     cfg = RobStrideConfig(channel='can0',
    #                           baud_rate=RSBaudRate.BPS_1M,
    #                           control_mode=RSRunMode.PP_POSITION_MODE,
    #                           host_can_id=0xfe,
    #                           motor_can_id=[0x7f],
    #                           pos_kp=None,
    #                           default_accel_PP_mode=None,
    #                           default_vel_PP_mode=None,
    #                           init_target_pos=None)
    #
    #     controller = RobStrideController(cfg)
    #     loop = asyncio.get_running_loop()
    #
    #     async with asyncio.TaskGroup() as tg:
    #         # task_io = tg.create_task(controller.send_rcv_task())
    #         task_io = tg.create_task()
    #         task_cpu_bound_policy = tg.create_task(mock_cpu_bound_policy(ctrl=controller))
    #
    #     print(f'==== finish task group ===')

    # asyncio.run(_main())