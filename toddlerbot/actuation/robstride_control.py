"""
experimented for robstride RS02 actuator.  by kenneth yu.
"""

import time
from typing import (Dict, NamedTuple, Sequence,
                    Tuple, Callable, List)
import numpy as np
import numpy.typing as npt
import can
from copy import deepcopy
# from aioconsole import aprint
import multiprocessing as mp
from queue import Full
import bisect

from toddlerbot.actuation.robstride_client import RSBaudRate,RSRunMode
from toddlerbot.actuation._module_logger import logger
from toddlerbot.actuation.base_controller import BaseController,JointState
from toddlerbot.actuation.robstride_sdk import *


class ControlMsg(NamedTuple):
    time_to_send: float     #in perf count.
    msg_list: List[can.Message]

class RobStrideConfig(NamedTuple):
    channel: str
    baud_rate: RSBaudRate
    run_mode: RSRunMode
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

    # send to io_proc
    _ctrl_msg_q: mp.Queue[ControlMsg]

    # recv from io_proc
    _motor_state_frame_q: mp.Queue[MotorStateFrame]    # for periodic motor state report.
    _motor_param_value_q: mp.Queue[SingleParamValue]   # for reading param tabel value.

    def __init__(self,*,
                 config: RobStrideConfig,
                 ctrl_msg_q:mp.Queue[ControlMsg],
                 motor_state_frame_q: mp.Queue[MotorStateFrame],
                 motor_param_value_q: mp.Queue[SingleParamValue]):
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
        # client: RobStrideIOProc
        _motor_ids: Tuple[int]

        logger.info(f'init robstride controller with target motor ids: {config.motor_can_id}'
                    f'\n with config: {config} ')

        self.config = config
        # NOTE: the index in self._motor_ids is used for read data array index, like pos,vel,etc.
        # we use immutable tuple instead of set/list.
        # and the element order is important, so we use sorted tuple to keep motor ids.
        self._motor_can_id = tuple( sorted(set(config.motor_can_id)) )

        if len(self._motor_can_id) != len(config.motor_can_id):
            raise ValueError(f'input config motor_can_id include duplicated values: {config.motor_can_id=:}')

        assert np.all(0 < np.asarray(self._motor_can_id)) and np.all(np.asarray(self._motor_can_id) <= 0x7f)
        self._set_of_motor_can_id = set(self._motor_can_id)

        assert 0x7f < config.host_can_id <= 0xfe
        self._host_can_id = config.host_can_id

        self._ctrl_msg_q = ctrl_msg_q
        self._motor_state_frame_q = motor_state_frame_q
        self._motor_param_value_q = motor_param_value_q

        # self.lock = Lock()

        # self.client = RobStrideIOProc(motor_can_id=self._motor_id,
        #                               host_can_id=config.host_can_id,
        #                               channel=config.channel,
        #                               baud_rate=config.baud_rate,
        #                               )

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
    # async def send_rcv_task(self):
    #     await self.client.send_rcv_task()

    @staticmethod
    def _set_param_with_double_check(*, set_fn:Callable[[Sequence[float|int] |int|float],None],
                                     set_value: Sequence[float|int] |float|int|None,
                                     read_tx_fn:Callable[[],None],
                                     get_fn:Callable[[], Sequence[int|float]],
                                     wait_sec:float):
        set_fn(set_value)
        # double check:
        read_tx_fn()
        time.sleep(wait_sec)
        fetched_value = get_fn()
        logger.info(f"fetched value from motors: {fetched_value}")
        if np.any(np.asarray(fetched_value) != set_value):
            raise IOError(
                f"not all motors are set through: {set_fn.__name__} to value: {set_value}."
            )

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

        # naive solution: method to wait 2. seconds for I/O task starting.
        time.sleep(2.)
        read_tx_and_get_value_interval_sec: float = 0.5

        logger.info(f'--- checking motor voltage --->')
        self.read_voltage_tx()
        time.sleep(read_tx_and_get_value_interval_sec)
        v_in = self.get_voltage_nowait()
        assert len(v_in)==len(self._motor_can_id)
        logger.info(f"read Voltage of motors: (V): {v_in}")
        if np.any(np.asarray(v_in,dtype=np.float32) < 46.):
            raise ValueError(
                "Voltage too low. Please check the power supply or charge the batteries."
            )

        # ---- TODO: add overload protect, min/max pos.. to RS motors. ----
        # set canTimeout.
        # self.set_return_delay_time(self.config.return_delay_us)

        logger.info(f'--- checking motor run mode --->')
        # naive solution: unify one mode for all the motors.
        run_mode_cmd: int = self.config.run_mode.convert_to_rs_cmd()
        assert run_mode_cmd in {RunModeCmd.MOTION,
                                RunModeCmd.PP_POSITION,
                                RunModeCmd.SPEED,
                                RunModeCmd.CURRENT,
                                RunModeCmd.CSP_POSITION,}

        self._set_param_with_double_check(set_fn=self.set_run_mode_nowait,
                                          set_value=run_mode_cmd,
                                          read_tx_fn=self.read_run_mode_tx,
                                          get_fn=self.get_run_mode_nowait,
                                          wait_sec = read_tx_and_get_value_interval_sec)

        assert (np.all(np.asarray(self.config.pos_kp, dtype=np.float32) <= ParamThreshold.KP_MAX)
                and np.all(0 < np.asarray(self.config.pos_kp, dtype=np.float32)))
        self._set_param_with_double_check(set_fn=self.set_pos_kp_nowait,
                                          set_value=self.config.pos_kp,
                                          read_tx_fn=self.read_pos_kp_tx,
                                          get_fn=self.get_pos_kp_nowait,
                                          wait_sec=read_tx_and_get_value_interval_sec)

        # TODO: check protection mode:

        # TODO:
        # check torque limit: EEPROM-16 and SRAM-48
        # check overload torque threshold/protection-duration/protection-torque:  EEPROM-34/35/36

        # set acc, vel, adjust present pos as init_pos from config.
        self.set_goal_accel(motor_ids=self._motor_can_id, accel=self.config.default_accel)
        self.set_goal_vel(motor_ids=self._motor_can_id, vel=self.config.default_vel)

        # TODO: temply set to 90% for sysID.
        # self.set_torque_limit(motor_ids=self._motor_id, limit_percentage=self.config.default_torque_limit)

        # NOTE: first set goal pos to init_pos in config.json, then normalize init pos read from motor.
        self.set_goal_pos(motor_ids=self._motor_can_id, pos=self.config.init_goal_pos)

        self.set_torque_enabled(motor_ids=self._motor_can_id, enabled=True)

        # NOTE: TO adjust the init pos bias: first set goal pos to init_pos in config.json,
        # then normalize init pos read from motor.
        # if config.init_pos is None, that is for calibrate_zero.
        # TODO: during calibrate_zero , setting init_pos to pi ??
        self._normalized_init_pos: npt.NDArray[np.float32] | None = None

        if self.config.init_target_pos is None:
            # for calibrate_zero.
            self._normalized_init_pos = np.zeros(len(self._motor_can_id), dtype=np.float32)
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
        _, read_pos = self.read_pos(retries=-1)
        # delta_pos = read_pos - self.normalized_init_pos

        delta_pos = read_pos - np.asarray(self.config.init_goal_pos, dtype=np.float32)

        delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi

        self._normalized_init_pos = read_pos - delta_pos

        assert np.all(abs(self._normalized_init_pos) <= np.pi)

        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')


    def close_motors(self):
        """Closes all active motor clients.

        This method iterates over all currently open Feite clients and forces them to close if they are in use. It logs a message for each client that is being forcibly closed and then sets the client's port handler to not in use before disconnecting the client.
        """
        open_clients: Set[FeiteGroupClient] = RobStrideIOProc.OPEN_CLIENTS  # type: ignore
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
        self.set_kp(kp)


    # NOTE: will offset using self.normalized_init_pos
    def set_pos(self, pos: Sequence[float]):
        """Sets the position of the motors by updating the desired position.

        Args:
            pos (Sequence): A list of position values to set for the motors.
        """

        # TODO: convert from [-pi/2, pi/2] to [pi/2, 3pi/2] -> during calibrate_zero , setting init_pos to pi...

        pos_arr: npt.NDArray[np.float32] = np.array(pos)
        # add init_pos as offset.
        pos_arr_drive = self._normalized_init_pos + pos_arr


        self.set_goal_pos(motor_ids=self._motor_can_id, pos=pos_arr_drive)



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

        read_value = self.read_pos_vel_load(retries=retries)


        assert len(self._motor_can_id) == len(read_value.pos) == len(read_value.vel) == len(read_value.load)

        # relative to init pos.
        relative_pos = read_value.pos - self._normalized_init_pos

        for _id, _pos, _vel, _load in zip(self._motor_can_id,
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

    def _tx_msg(self, msg:List[can.Message],
                time_to_send:float = 0.):
        """
        dump can messages to IO task.
        Args:
            msg: List of can messages.
            time_to_send:in perf count. tell IOTask postpone to future time for sending this group of can messages.
                         default: 0., means send immediately.
        """
        try:
            # for _m in msg:
            #     # will raise immediately if full.
            #     self._ctrl_msg_q.put_nowait(_m)

            # raise immediately if full.
            ctrl_msg = ControlMsg(time_to_send=time_to_send, msg_list=msg)
            self._ctrl_msg_q.put_nowait(ctrl_msg)

        except Full as exc:
            logger.error(f'tx msg failed: _ctrl_msg_q is full ---> '
                             f'length:{self._ctrl_msg_q.qsize()}.'
                             f' check the application running freq and asyncio send bandwidth.'
                         f'{exc=:} {type(exc)=:}')
            raise exc

    # executed in run_policy process. not real time data, we can set wait time.
    def _read_param_tx_helper(self, name:str)->None:

        if name not in RS_param_table_spec:
            raise KeyError(f'read param tx failed, param name:{name} not in RS_param_table_spec. ')

        index = RS_param_table_spec[name].index

        # build read msg.
        snd_msg:List[can.Message] = RSProtocolBuilder.read_single_param(motor_can_id=self._motor_can_id,
                                            host_can_id=self._host_can_id,
                                            index=index)
        self._tx_msg(snd_msg)


    def _write_param_tx_helper(self, name:str, value:Sequence[float|int])->None:

        if name not in RS_param_table_spec:
            raise KeyError(f'write param tx failed, param name:{name} not in RS_param_table_spec. ')

        p_spec:ParamSpec = RS_param_table_spec[name]

        # build read msg.
        snd_msg: List[can.Message] = RSProtocolBuilder.write_single_param(motor_can_id=self._motor_can_id,
                                                                          host_can_id=self._host_can_id,
                                                                          index=p_spec.index,
                                                                          param_value=value,
                                                                          param_spec=p_spec,
                                                                          )
        self._tx_msg(snd_msg)


    # blocking get.
    def _get_motor_state_helper(self, timeout_sec:float) \
            ->Dict[int,JointState]:  #  List[SingleParamValue]:
        """
         raise exc if the corresponding param not received.
        """
        # TODO: guarantee the order.

        # TODO: ordered dict?
        state_dict: Dict[int, JointState] = {}
        # rcv_motor_id:set[int] = set()

        timeout_point: float = time.perf_counter() + timeout_sec

        try:
            # TODO: naive solution.
            while len(state_dict) < len(self._set_of_motor_can_id):
                while not self._motor_state_frame_q.empty():
                    state:MotorStateFrame = self._motor_state_frame_q.get_nowait()

                    # TODO: check obsolete frame, waiting coming frame...
                    assert state.can_id in self._set_of_motor_can_id
                    # less than 10ms
                    assert time.time() - state.ts < 1e-2

                    # assert state.can_id not in rcv_motor_id
                    # rcv_motor_id.add(state.can_id)
                    assert state.can_id not in state_dict

                    # relative to init pos.
                    relative_pos = state.pos - self._normalized_init_pos

                    state_dict[state.can_id] = JointState(time=state.ts,
                                                          pos=relative_pos,
                                                          vel=state.vel,
                                                          tor=state.torque,
                                                          temp=state.temp)

                    # if len(state_dict) == len(self._set_of_motor_can_id) \
                    #     and state_dict.keys() == self._set_of_motor_can_id:
                    #     logger.debug(f'get motor state frame from all the motor.')
                    #     break

                # checkout timeout:
                if time.perf_counter() > timeout_point:
                    raise IOError(f'get motor state timeout, timeout sec:{timeout_sec}')

                # yield to wait queueing.
                time.sleep(0.)

            assert state_dict.keys() == self._set_of_motor_can_id
            return state_dict

        except Exception as exc:
            logger.error(f'get motor state helper failed: {exc=:} {type(exc)=:}')
            raise exc


    # blocking get.
    def _get_param_value_helper(self, name: str, timeout_sec:float) \
            -> npt.NDArray[np.float32|np.int32]:  #  List[SingleParamValue]:
        """
         raise exc if the corresponding param not received.
        """
        # TODO: guarantee the order.

        if name not in RS_param_table_spec:
            raise KeyError(f'get param failed, param name:{name} not in RS_param_table_spec. ')

        index = RS_param_table_spec[name].index
        # ret: List[SingleParamValue] = list()
        if RS_param_table_spec[name].dtype is int:
            dtype = np.uint32
        elif RS_param_table_spec[name].dtype is float:
            dtype = np.float32
        else:
            raise TypeError

        value_arr = np.empty(shape=len(self._motor_can_id), dtype=dtype)
        rcv_motor_id:set[int] = set()

        deadline:float = time.perf_counter() + timeout_sec

        try:
            # TODO: naive solution.
            # while not self._motor_param_value_q.empty():
            while len(rcv_motor_id) < len(self._set_of_motor_can_id):
                # while not self._motor_param_value_q.empty():
                # if not self._motor_param_value_q.empty():
                # value:SingleParamValue = self._motor_param_value_q.get_nowait()

                # check timeout:
                q_get_timeout:float = deadline - time.perf_counter()
                if q_get_timeout < 0:
                    raise IOError(f'get param value timeout, timeout sec:{timeout_sec}')

                # mp.Queue will raise Empty if timeout.
                value: SingleParamValue = self._motor_param_value_q.get(block=True,
                                                                        timeout=q_get_timeout)

                assert value is not None
                # TODO: maybe cache the param value if not wanted index/repeated_motor_id.
                assert value.index == index
                assert value.can_id in self._set_of_motor_can_id
                # less than 10ms
                assert time.time() - value.ts < 1e-2

                assert value.can_id not in rcv_motor_id
                rcv_motor_id.add(value.can_id)

                # TODO: use same order as in _motor_can_id...
                # TODO: use heapq to optimize index.
                # insert_idx: int = self._motor_can_id.index(value.can_id)
                # NOTE: self._motor_can_id must be sorted.
                insert_idx: int = bisect.bisect_left(self._motor_can_id,x=value.can_id)
                value_arr[insert_idx] = value.value

                # if len(rcv_motor_id) == len(self._set_of_motor_can_id) \
                #     and rcv_motor_id == self._set_of_motor_can_id:
                #     logger.debug(f'get param value from all the motor. index:{index}')
                #     break

                # # checkout timeout:
                # if time.perf_counter() > deadline:
                #     raise IOError(f'get param value timeout, timeout sec:{timeout_sec}')

                # yield to wait queueing.
                # time.sleep(0.)

            assert rcv_motor_id == self._set_of_motor_can_id
            return value_arr

        except Exception as exc:
            logger.error(f'get param value failed: {exc=:} {type(exc)=:}')
            raise exc


    # def _get_param_value_helper(self, name:str)->List[SingleParamValue]:
    #     """
    #      raise exc if the corresponding param not received.
    #     """
    #     # TODO: guarantee the order.
    #
    #     if name not in RS_param_table_spec:
    #         raise KeyError(f'get param failed, param name:{name} not in RS_param_table_spec. ')
    #
    #     index = RS_param_table_spec[name].index
    #     ret:List[SingleParamValue] = []
    #
    #     for _id in self._motor_can_id:
    #
    #         print(f'+++ {_id=:} {self._motor_param_table=:}')
    #
    #         p_table = self._motor_param_table[_id]
    #         value = p_table[index]
    #         if value is None:
    #             raise ValueError(f'read param table failed, the value is None which means'
    #                              f'the corresponding motor does not feedback read param: '
    #                              f'index: 0x{index:x}, motor id: {_id} ')
    #         ret.append(deepcopy(value))
    #         # clear the cached data.
    #         p_table[index] = None
    #
    #     return ret

        # non block.

    # def get_motor_state_nowait(self) -> List[MotorStateFrame]:
    #     """
    #     return list containing state frame of all the motors.
    #     """
    #
    #     # TODO: guarantee the order.
    #
    #     motor_state: List[MotorStateFrame] = []
    #     for _id in self._motor_can_id:
    #         state_q = self._motor_state_q[_id]
    #         if len(state_q) == 0:
    #             raise ValueError(f'motor state deque is empty, motor can id:{_id}. '
    #                              f'check the corresponding motor +48V supply and can bus connection.')
    #
    #         # Remove and return the rightmost element which is latest?
    #         # TODO: pop() is atomic?
    #         # motor_state.append(state_q.pop())
    #         motor_state.append(deepcopy(state_q.popleft()))
    #         # state_q.clear()
    #     return motor_state


    # TODO: return timestamp.
    def get_voltage_nowait(self)->List[float]:
        param:List[SingleParamValue] = self._get_param_value_helper('VBUS',0)
        return [_p.value for _p in param]

    def read_voltage_tx(self)->None:
        self._read_param_tx_helper('VBUS')

    # TODO: return timestamp.
    def get_run_mode_nowait(self) -> List[int]:
        param: List[SingleParamValue] = self._get_param_value_helper('run_mode')
        return [_p.value for _p in param]

    def read_run_mode_tx(self)->None:
        self._read_param_tx_helper('run_mode')

    def get_pos_kp_nowait(self) -> List[float]:
        param: List[SingleParamValue] = self._get_param_value_helper('loc_kp')
        return [_p.value for _p in param]

    def read_pos_kp_tx(self)->None:
        self._read_param_tx_helper('loc_kp')

    def set_target_accel_nowait(self, accel: npt.NDArray[np.float32])->None:
        raise NotImplemented

    def set_target_vel_nowait(self, vel: npt.NDArray[np.float32])->None:
        raise NotImplemented

    def set_target_pos_nowait(self, pos: npt.NDArray[np.float32])->None:
        """Writes the given desired positions.

        Args:
            pos: The joint angles in radians to write. in rad of single turn.signed value,
             to represent rotor direction.
             element order in `pos` must be same as self.motor_can_id.
        """
        assert len(self._motor_can_id) == len(pos)
        # TODO: only allow -2Pi ~ 2Pi.
        if not np.all(np.abs(pos) < 2 * np.pi):
            raise ValueError(f'not allowed goal pos: {pos}, which should be in [-2pi, 2pi] ')

        self._write_param_tx_helper('loc_ref', pos)


    def set_run_mode_nowait(self, mode: int)->None:
        assert mode in {RunModeCmd.MOTION, RunModeCmd.PP_POSITION,
                        RunModeCmd.SPEED, RunModeCmd.CURRENT,
                        RunModeCmd.CSP_POSITION}

        self._write_param_tx_helper('run_mode', [mode]*len(self._motor_can_id) )

    def set_pos_kp_nowait(self, kp: Sequence[float])->None:
        self._write_param_tx_helper('loc_kp', kp)

    def set_mech_zero_nowait(self)->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.set_mech_pos_zero(motor_can_id=self._motor_can_id,
                                                                        host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def set_motor_enable_nowait(self)->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.motor_enable(motor_can_id=self._motor_can_id,
                                                                   host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def set_motor_disable_nowait(self)->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.motor_disable(motor_can_id=self._motor_can_id,
                                                                   host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def read_model_number_tx(self, wait_sec: float)->None:
        snd_msg: List[can.Message] = RSProtocolBuilder.get_device_id(motor_can_id=self._motor_can_id,
                                                                    host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)



if __name__ == '__main__':
    # import concurrent.futures
    import asyncio
    # import threading
    import atexit

    def mock_cpu_bound_policy(ctrl: BaseController):
        time.sleep(2.0)
        print(f'---> start initialize motors')
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
                          run_mode=RSRunMode.PP_POSITION,
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

    # io_thrd = threading.Thread(target=run_io_bound_task_in_spawned_process,args=[controller],daemon=True)

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

    # def clean_io_process():
    #     print(f'clean_io_process(): terminate io_process gracefully.')
    #     while io_thrd.is_alive():
    #         print(f'io_proc is still alive: {io_thrd.is_alive()}, terminate it')
    #         io_thrd.terminate()
    #         time.sleep(0.5)
    #     # fut.cancel()
    #     print(f'io_proc is alive: {io_thrd.is_alive()}')
    #     print(f'close motors:')
    #     controller.close_motors()
    #     time.sleep(0.5)

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
        # TODO: naive solution to wait for the io task running. maybe using connection?
        while not io_proc.is_alive():
            time.sleep(0.5)

        mock_cpu_bound_policy(controller)

    # try:
    #     print(f'start io_process.')
    #     io_thrd.start()
    #     # io_proc.join()
    #     print(f'io_proc is alive: {io_thrd.is_alive()}')
    #     print(f'start mock cpu bound policy.')
    #     # TODO: naive solution to wait for the io task running. maybe using connection?
    #     while not io_thrd.is_alive():
    #         time.sleep(0.5)
    #
    #     mock_cpu_bound_policy(controller)


    except Exception as error:
        print(f'--- exception in main process: {error=:} {type(error)=:}')
        time.sleep(0.5)
        raise error

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