"""
experimented for robstride RS02 actuator.  by kenneth yu.
"""

import time
from typing import (Dict, NamedTuple, Sequence,
                    Tuple, Callable, List, OrderedDict)
import numpy as np
import numpy.typing as npt
import can
# from aioconsole import aprint
import multiprocessing as mp
from multiprocessing.connection import Connection
from queue import Full
# import bisect
from functools import partial
import logging

if __name__ == '__main__':
    from toddlerbot.actuation.robstride_io_proc import RSBaudRate,RSRunMode, RSReportPeriod, ControlMsg, RSIOEvent
    from toddlerbot.actuation._module_logger import logger
    from toddlerbot.actuation.base_controller import BaseController,JointState
    from toddlerbot.actuation.robstride_sdk import *
    from toddlerbot.utils import config_logging
else:
    from .robstride_io_proc import RSBaudRate, RSRunMode, RSReportPeriod, ControlMsg, RSIOEvent
    from ._module_logger import logger
    from .base_controller import BaseController, JointState
    from .robstride_sdk import *
    from ..utils import config_logging


class RobStrideConfig(NamedTuple):
    channel: str
    baud_rate: RSBaudRate
    run_mode: RSRunMode
    motor_report_period: RSReportPeriod
    motor_can_id: Sequence[int]
    host_can_id: int
    pos_kp: Sequence[float] | None = None
    # vel_kp: Sequence[int] | None
    # vel_ki: Sequence[int] | None
    # TODO: adjust according to tracking result.
    # TODO: move into config.json.
    # default_torque_limit: npt.NDArray[np.uint16] |None = None

    default_accel_PP_mode: Sequence[float] |None = None   # [1.6 * np.pi]
    default_vel_PP_mode: Sequence[float] |None = None      # [1.4 * np.pi]
    init_target_pos: Sequence[float] |None = None    # None = None

    # interp_method: str = "cubic"


class RobStrideController(BaseController):
    """Class for controlling RobStride RS02 motors."""

    # events between with io_proc
    _event_conn_with_io_proc: Connection
    # send to io_proc
    _motor_ctrl_q: mp.Queue  #[ControlMsg]
    # recv from io_proc
    _motor_state_frame_q: mp.Queue #[MotorStateFrame]    # for periodic motor state report.
    _motor_param_value_q: mp.Queue #[SingleParamValue]   # for reading param tabel value.

    # index in self._motor_can_id tuple.
    _can_id_to_ordering_index: OrderedDict[int, int]

    def __init__(self, *,
                 config: RobStrideConfig,
                 event_conn_with_io_proc: Connection,
                 motor_ctrl_q:mp.Queue,  #[ControlMsg],
                 motor_state_frame_q: mp.Queue,  # [MotorStateFrame],
                 motor_param_value_q: mp.Queue,  # [SingleParamValue]
                 ):
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
        self._motor_can_id_ordering = tuple(sorted(set(config.motor_can_id)))

        if len(self._motor_can_id_ordering) != len(config.motor_can_id):
            raise ValueError(f'input config motor_can_id include duplicated values: {config.motor_can_id=:}')

        assert np.all(0 < np.asarray(self._motor_can_id_ordering)) and np.all(np.asarray(self._motor_can_id_ordering) <= 0x7f)

        # self._can_id_to_ordering_index:OrderedDict[int,int] = OrderedDict( enumerate(self._motor_can_id_ordering) )
        self._can_id_to_ordering_index: OrderedDict[int, int] = OrderedDict(zip(self._motor_can_id_ordering,
                                                                                range(len(self._motor_can_id_ordering))))

        self._set_of_motor_can_id = set(self._motor_can_id_ordering)

        assert 0x7f < config.host_can_id <= 0xfe
        self._host_can_id = config.host_can_id

        self._event_conn_with_io_proc = event_conn_with_io_proc
        self._motor_ctrl_q = motor_ctrl_q
        self._motor_state_frame_q = motor_state_frame_q
        self._motor_param_value_q = motor_param_value_q

        # self.lock = Lock()

        # self.client = RobStrideIOProc(motor_can_id=self._motor_id,
        #                               host_can_id=config.host_can_id,
        #                               channel=config.channel,
        #                               baud_rate=config.baud_rate,
        #                               )

        # self.initialize_motors()

        # NOTE: TO adjust the init pos bias: first set goal pos to init_pos in config.json,
        # then normalize init pos read from motor.
        # if config.init_pos is None, that is for calibrate_zero.
        # TODO: during calibrate_zero , setting init_pos to pi ??
        # self._normalized_init_pos: npt.NDArray[np.float32] | None = None
        # init to zero:
        self._normalized_init_pos : npt.NDArray[np.float32] = np.zeros(len(self._motor_can_id_ordering), dtype=np.float32)

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
                                     read_tx_fn:Callable[[],None] | None,
                                     get_fn:Callable[[], Sequence[int|float]]
                                     ):
        set_fn(set_value)

        # double check:
        if read_tx_fn is not None:
            read_tx_fn()
            # time.sleep(wait_sec)
            fetched_value = get_fn()
            logger.info(f"fetched value from motors: {fetched_value}")
            if np.any(np.asarray(fetched_value) != set_value):
                raise IOError(
                    f"not all motors are set through: {set_fn.__name__} to value: {set_value}."
                )


    # NOTE: called after send_rcv_task running in loop.
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

        # waiting for I/O task ready.
        logger.warning("=== waiting for RSIOEvent.ready ===")
        event = self._event_conn_with_io_proc.recv()
        logger.warning(f" rcv event from io proc: {event}")
        if event is RSIOEvent.SndRcvTaskReady:
            logger.warning("=== get RSIOEvent.ready, start send initializing msg to io proc. ===")
        else:
            # TODO: naive implementation. using state_machine in future.
            raise NotImplementedError(f'we only support RSIOEvent.Ready event in controller initialize_motors state. '
                                      f'TODO: using state_machine.')

        get_timeout_sec: float = 0.5

        logger.info(f'=== checking motor voltage ===')
        self.read_voltage_tx()
        v_in = self.get_voltage(get_timeout_sec)
        assert len(v_in)==len(self._motor_can_id_ordering)
        logger.info(f"read Voltage of motors: (V): {v_in}")
        if np.any(np.asarray(v_in,dtype=np.float32) < 46.) or np.any(np.asarray(v_in,dtype=np.float32) > 50.):
            raise ValueError(
                "Voltage too low or too high than +48V."
                " Please check the power supply or charge the batteries."
            )

        # ---- TODO: add overload protect, min/max pos.. to RS motors. ----
        # set canTimeout.
        # self.set_return_delay_time(self.config.return_delay_us)

        # NOTE: run mode can only be set in `motor-disabled` state.
        logger.info(f'=== set and check motor run mode ===')
        self._set_and_check_run_mode(run_mode=self.config.run_mode,
                                     get_timeout_sec=get_timeout_sec)


        if self.config.pos_kp is not None:
            logger.info(f'=== set and check motor pos kp ===')
            self._set_and_check_pos_kp(kp=self.config.pos_kp,
                                       get_timeout_sec=get_timeout_sec)

        # TODO: check protection mode:
        # TODO: check torque limit:
        # self.set_torque_limit(motor_ids=self._motor_id, limit_percentage=self.config.default_torque_limit)

        # TODO: we only use PP mode, to set accel/vel for PP mode.
        # set acc, vel, adjust present pos as init_pos from config.
        if self.config.default_accel_PP_mode is not None:
            self.set_target_accel_nowait(np.asarray(self.config.default_accel_PP_mode, dtype=np.float32))
            time.sleep(0.1)

        if self.config.default_vel_PP_mode is not None:
            self.set_target_vel_nowait(np.asarray(self.config.default_vel_PP_mode, dtype=np.float32))
            time.sleep(0.1)

        logger.info(f'=== set mech pos zero scope to -pi ~ pi ===')
        # TODO: for RS03/04, the 0x7026~0x7029 params are write only, not readable.
        # set mech zero scope to -pi~pi.
        self._set_and_check_zero_scope(in_neg_pi_pos_pi=True,
                                       get_timeout_sec=get_timeout_sec,
                                       # TODO: distinguish RS02/03/04
                                       read_and_check=False)

        logger.info(f'=== set mech pos zero ===')
        # TODO: check mech pos?
        self.set_mech_pos_zero_nowait()

        logger.warning(f'=== enable all the motors ===')
        # TODO: how to check all the motors are enabled?
        # while not self._motor_state_frame_q.empty():
        #     _ste = self._motor_state_frame_q.get_nowait()
        #     logger.warning(f'\n--- flush motor state: {_ste} ---')
        #     time.sleep(0.2)

        self._enable_motor_nowait(self._motor_can_id_ordering)
        logger.warning(f'=== read curr mech pos and wait for the param feedback to guarantee motor enabled.')
        # wait for RS motor ready
        time.sleep(3.)
        self.read_mech_pos_tx()
        curr_pos = self.get_mech_pos(timeout_sec=0.5)
        logger.warning(f'=== get curr_pos: {curr_pos}')
        logger.warning(f'=== motor enabled successfully ===')

        # logger.info(f'--- set and normalize motor init pos  --->')
        # # NOTE: first set goal pos to init_pos in config.json, then normalize init pos read from motor.
        # if self.config.init_target_pos is not None:
        #     self.set_target_pos_nowait(self.config.init_target_pos)

        logger.info(f'=== set and normalize motor init pos  ===')
        self._set_and_normalize_init_pos()

        # if self.config.init_target_pos is None:
        #     for calibrate_zero.
            # self._normalized_init_pos = np.zeros(len(self._motor_can_id), dtype=np.float32)
        # else:
        #     self._normalize_init_pos()

        logger.info(f'=== set and check motor state report period ===')
        # TODO: for RS03/04, the 0x7026~0x7029 params are write only, not readable.
        self._set_and_check_motor_report_period(period=self.config.motor_report_period,
                                                get_timeout_sec=get_timeout_sec,
                                                # TODO: distinguish RS02/03/04.
                                                read_and_check=False)

        time.sleep(2.)

        # self.set_target_pos_nowait(np.asarray([0.87], dtype=np.float32))
        # time.sleep(2.)
        # self.set_target_pos_nowait(np.asarray([0.99], dtype=np.float32))
        # time.sleep(2.)

        # logger.info(f'=== enable the motor state periodic report  ===')
        # TODO: temply for checking the report period.
        # self.toggle_periodic_report_nowait(enable=True)
        # time.sleep(1.0)
        # logger.info(f'=== disable the motor state periodic report  ===')
        # self.toggle_periodic_report_nowait(enable=False)
        # time.sleep(1.0)


    # TODO: for RS04, the 0x7026~0x7029 params are write only, not readable.
    def _set_and_check_zero_scope(self, *, in_neg_pi_pos_pi:bool, get_timeout_sec:float, read_and_check:bool):
        # cmd: 0~2pi:0,  -pi~pi: 1
        scope_cmd:int = 1 if in_neg_pi_pos_pi else 0
        self._set_param_with_double_check(set_fn=self.set_zero_scope_nowait,
                                          set_value=scope_cmd,
                                          read_tx_fn=self.read_zero_scope_tx if read_and_check else None,
                                          get_fn=partial(self.get_zero_scope, get_timeout_sec)
                                          )

    def _set_and_check_run_mode(self, run_mode:RSRunMode, get_timeout_sec:float)->None:
        # naive solution: unify one mode for all the motors.
        run_mode_cmd: int = run_mode.convert_to_rs_cmd()
        assert run_mode_cmd in {RunModeCmd.MOTION,
                                RunModeCmd.PP_POSITION,
                                RunModeCmd.SPEED,
                                RunModeCmd.CURRENT,
                                RunModeCmd.CSP_POSITION, }

        self._set_param_with_double_check(set_fn=self.set_run_mode_nowait,
                                          set_value=run_mode_cmd,
                                          read_tx_fn=self.read_run_mode_tx,
                                          get_fn=partial(self.get_run_mode, get_timeout_sec)
                                          )

    def _set_and_check_motor_report_period(self, *, period: RSReportPeriod,
                                           get_timeout_sec:float,
                                           read_and_check:bool):
        period_cmd: int = period.convert_to_rs_cmd()
        assert period_cmd in {ReportPeriodCmd.P_10MS,
                              ReportPeriodCmd.P_15MS,
                              ReportPeriodCmd.P_20MS,
                              ReportPeriodCmd.P_25MS,
                              ReportPeriodCmd.P_30MS,
                              ReportPeriodCmd.P_35MS,
                              ReportPeriodCmd.P_40MS,
                              ReportPeriodCmd.P_45MS,
                              }

        self._set_param_with_double_check(set_fn=self.set_motor_report_period_nowait,
                                          set_value=period_cmd,
                                          read_tx_fn=self.read_motor_report_period_tx if read_and_check else None,
                                          get_fn=partial(self.get_motor_report_period,
                                                         get_timeout_sec)
                                          )

    def _set_and_check_pos_kp(self, kp: Sequence[float | int], get_timeout_sec: float):
        """
        for PP mode.
        """
        assert (np.all(np.asarray(kp, dtype=np.float32) <= ParamThreshold.KP_MAX)
                and np.all(0 < np.asarray(kp, dtype=np.float32)))
        self._set_param_with_double_check(set_fn=self.set_pos_kp_nowait,
                                          set_value=kp,
                                          read_tx_fn=self.read_pos_kp_tx,
                                          get_fn=partial(self.get_pos_kp, get_timeout_sec)
                                          )

    def _set_and_normalize_init_pos(self):
        """Update the initial position to account for any changes in position.

        This method reads the current position from the client and calculates the
        difference from the stored initial position. It then adjusts the initial
        position to reflect any changes, ensuring that the position remains within
        the range of [-π, π].
        """

        # NOTE: first set goal pos to init_pos in config.json, then normalize init pos read from motor.
        if self.config.init_target_pos is not None:
            self.set_target_pos_nowait(np.asarray(self.config.init_target_pos,dtype=np.float32))

            self.read_mech_pos_tx()
            curr_pos = self.get_mech_pos(timeout_sec=0.5)
            logger.warning(f'read curr_pos: {curr_pos}')

            # delta_pos = read_pos - self.normalized_init_pos

            delta_pos = curr_pos - np.asarray(self.config.init_target_pos, dtype=np.float32)

            # TODO: optimize if delta_pos is very tiny. set to _normalized_init_pos None.
            delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi

            self._normalized_init_pos = curr_pos - delta_pos

        else:
            # for calibrate_zero.
            self._normalized_init_pos = np.zeros(len(self._motor_can_id_ordering), dtype=np.float32)

        # -pi ~ pi
        assert np.all(abs(self._normalized_init_pos) <= np.pi)

        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')
        logger.warning(f'====== normalized init pos: {self._normalized_init_pos} =============')


    def close_motors(self):
        """Closes all active motor clients.

        This method iterates over all currently open Feite clients and forces them to close if they are in use. It logs a message for each client that is being forcibly closed and then sets the client's port handler to not in use before disconnecting the client.
        """

        logger.warning(f'=== close all the motors ===')
        open_clients: Set[FeiteGroupClient] = RobStrideIOProc.OPEN_CLIENTS  # type: ignore
        for _client in open_clients:
            # will tx motor_disable can msg to all motors:
            _client.disconnect()

    def set_kp(self, kp: Sequence[int|float]):
        raise NotImplementedError

    def set_pos_kp(self, kp: Sequence[int|float]):
        self._set_and_check_pos_kp(kp, get_timeout_sec=0.5)

    # NOTE: will offset using self.normalized_init_pos
    def set_pos(self, pos: Sequence[float]):
        """Sets the position of the motors by updating the desired position.

        Args:
            pos (Sequence): A list of position values to set for the motors.
        """

        # TODO: convert from [-pi/2, pi/2] to [pi/2, 3pi/2] -> during calibrate_zero , setting init_pos to pi...

        pos_arr: npt.NDArray[np.float32] = np.asarray(pos,dtype=np.float32)
        # add init_pos as offset.
        pos_arr_biased = self._normalized_init_pos + pos_arr

        # we do not wait for the feedback, and we use periodic motor state report to get the curr pos of motors.
        self.set_target_pos_nowait(pos_arr_biased)

    # NOTE: will offset using self.normalized_init_pos
    # @profile()
    def get_motor_state(self, timeout_sec:float|None) -> Dict[int, JointState]:
        """Retrieves the current state of the motors, including position, velocity, and current.

        Args:
            timeout_sec (int): The number of retry attempts for reading motor data in case of failure. Defaults to 0.

        Returns:
            Dict[int, JointState]: A dictionary mapping motor IDs to their respective `JointState`, which includes time, position, velocity, and torque.
        """

        # TODO: convert POS from [pi/2, 3pi/2] to [-pi/2, pi/2].

        # in 10ms.
        return self._get_motor_state_helper(timeout_sec)

        # assert len(self._motor_can_id) == len(state_dict)

        # already biased in _get_motor_state_helper
        # relative to init pos.
        # relative_pos = read_value.pos - self._normalized_init_pos

        # return state_dict

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
            self._motor_ctrl_q.put_nowait(ctrl_msg)

        except Full as exc:
            logger.error(f'tx msg failed: _ctrl_msg_q is full ---> '
                             f'length:{self._motor_ctrl_q.qsize()}.'
                             f' check the application running freq and asyncio send bandwidth.'
                         f'{exc=:} {type(exc)=:}')
            raise exc

        except Exception as other_exc:
            raise other_exc

    # executed in run_policy process. not real time data, we can set wait time.
    def _read_param_tx_helper(self, name:str)->None:

        if name not in RS_param_table_spec:
            raise KeyError(f'read param tx failed, param name:{name} not in RS_param_table_spec. ')

        index = RS_param_table_spec[name].index

        # build read msg.
        snd_msg:List[can.Message] = RSProtocolBuilder.read_single_param(motor_can_id=self._motor_can_id_ordering,
                                                                        host_can_id=self._host_can_id,
                                                                        index=index)
        self._tx_msg(snd_msg)


    def _write_param_tx_helper(self, name:str, value:Sequence[float|int])->None:

        if name not in RS_param_table_spec:
            raise KeyError(f'write param tx failed, param name:{name} not in RS_param_table_spec. ')

        p_spec:ParamSpec = RS_param_table_spec[name]

        # build read msg.
        snd_msg: List[can.Message] = RSProtocolBuilder.write_single_param(motor_can_id=self._motor_can_id_ordering,
                                                                          host_can_id=self._host_can_id,
                                                                          index=p_spec.index,
                                                                          param_value=value,
                                                                          param_spec=p_spec,
                                                                          )
        self._tx_msg(snd_msg)


    # blocking get.
    def _get_motor_state_helper(self, timeout_sec:float|None) \
            ->Dict[int,JointState]:  #  List[SingleParamValue]:
        """
         raise exc if the corresponding param not received.
         Args:
             timeout_sec: None -- wait forever.

        """
        # TODO: guarantee the order.

        # TODO: ordered dict?
        state_dict: Dict[int, JointState] = {}
        # rcv_motor_id:set[int] = set()

        deadline: float = -1.

        if timeout_sec is not None:
            deadline:float = time.perf_counter() + timeout_sec

        try:
            # TODO: naive solution.
            while len(state_dict) < len(self._set_of_motor_can_id):
                # check timeout:
                if timeout_sec is not None:
                    q_get_timeout: float|None = deadline - time.perf_counter()
                    if q_get_timeout < 0:
                        raise IOError(f'get motor state frame timeout, timeout sec:{timeout_sec}')
                else:
                    # block waiting forever.
                    q_get_timeout = None

                # mp.Queue will raise Empty if timeout.
                state:MotorStateFrame = self._motor_state_frame_q.get(block=True,timeout=q_get_timeout)

                assert state is not None
                # TODO: check obsolete frame, waiting coming frame...
                assert state.can_id in self._set_of_motor_can_id
                # less than 10ms
                assert time.time() - state.ts < 1e-2

                # assert state.can_id not in rcv_motor_id
                # rcv_motor_id.add(state.can_id)
                assert state.can_id not in state_dict

                # relative to init pos.
                ordering_idx = self._can_id_to_ordering_index[state.can_id]
                relative_pos = state.pos - self._normalized_init_pos[ordering_idx]

                state_dict[state.can_id] = JointState(time=state.ts,
                                                      pos=relative_pos,
                                                      vel=state.vel,
                                                      tor=state.torque,
                                                      temp=state.temp)

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

        value_arr = np.empty(shape=len(self._motor_can_id_ordering), dtype=dtype)
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
                # insert_idx: int = bisect.bisect_left(self._motor_can_id_ordering, x=value.can_id)
                ordering_idx = self._can_id_to_ordering_index[value.can_id]
                value_arr[ordering_idx] = value.value

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
    # def get_voltage_nowait(self)->List[float]:
    #     param:List[SingleParamValue] = self._get_param_value_helper('VBUS',0)
    #     return [_p.value for _p in param]

    def get_voltage(self, timeout_sec:float)->npt.NDArray[np.float32|np.int32]:
        return self._get_param_value_helper('VBUS', timeout_sec)

    def read_voltage_tx(self)->None:
        self._read_param_tx_helper('VBUS')


    def set_run_mode_nowait(self, mode_cmd: int) -> None:
        assert mode_cmd in {RunModeCmd.MOTION, RunModeCmd.PP_POSITION,
                            RunModeCmd.SPEED, RunModeCmd.CURRENT,
                            RunModeCmd.CSP_POSITION}

        self._write_param_tx_helper('run_mode', [mode_cmd] * len(self._motor_can_id_ordering))

    # TODO: return timestamp.
    def get_run_mode(self, timeout_sec: float) -> npt.NDArray[np.float32|np.int32]:
        return self._get_param_value_helper('run_mode', timeout_sec)

    def read_run_mode_tx(self)->None:
        self._read_param_tx_helper('run_mode')

    def set_pos_kp_nowait(self, kp: Sequence[float]) -> None:
        self._write_param_tx_helper('loc_kp', kp)

    def get_pos_kp(self, timeout_sec:float) -> npt.NDArray[np.float32|np.int32]:
        return self._get_param_value_helper('loc_kp', timeout_sec)

    def read_pos_kp_tx(self)->None:
        self._read_param_tx_helper('loc_kp')

    def set_mech_pos_zero_nowait(self)->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.set_mech_pos_zero(motor_can_id=self._motor_can_id_ordering,
                                                                        host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def set_zero_scope_nowait(self, cmd:int)->None:
        # cmd: 0~2pi:0,  -pi~pi: 1
        # cmd :int = 1 if in_neg_pi_pos_pi else 0
        assert cmd in {0,1}
        self._write_param_tx_helper(name='zero_sta', value=[cmd] * len(self._motor_can_id_ordering))

    def read_zero_scope_tx(self)->None:
        self._read_param_tx_helper('zero_sta')

    def get_zero_scope(self, timeout_sec:float):
        return self._get_param_value_helper('zero_sta', timeout_sec)

    def set_target_accel_nowait(self, accel: npt.NDArray[np.float32])->None:
        min_max =RS_param_table_spec['acc_set'].min_max
        assert np.all(min_max[0] < accel) and np.all( accel <= min_max[1])
        self._write_param_tx_helper(name='acc_set', value=accel)

    def set_target_vel_nowait(self, vel: npt.NDArray[np.float32])->None:
        min_max =RS_param_table_spec['vel_max'].min_max
        assert np.all(min_max[0] < vel) and np.all( vel <= min_max[1])
        self._write_param_tx_helper(name='vel_max', value=vel)

    def set_target_pos_nowait(self, pos: npt.NDArray[np.float32])->None:
        """Writes the given desired positions.

        Args:
            pos: The joint angles in radians to write. in rad of single turn.signed value,
             to represent rotor direction.
             element order in `pos` must be same as self.motor_can_id.
        """
        assert len(self._motor_can_id_ordering) == len(pos)
        # TODO: only allow -2Pi ~ 2Pi.
        if not np.all(np.abs(pos) < 2 * np.pi):
            raise ValueError(f'not allowed goal pos: {pos}, which should be in [-2pi, 2pi] ')

        self._write_param_tx_helper('loc_ref', pos)

    def read_mech_pos_tx(self)->None:
        self._read_param_tx_helper('mechPos')

    def get_mech_pos(self, timeout_sec:float)->npt.NDArray[np.float32|np.int32]:
        return self._get_param_value_helper('mechPos', timeout_sec)

    def toggle_periodic_report_nowait(self, enable:bool):
        snd_msg: List[can.Message] = RSProtocolBuilder.toggle_motor_periodic_report(motor_can_id=self._motor_can_id_ordering,
                                                                                    host_can_id=self._host_can_id, enable=enable)
        self._tx_msg(snd_msg)

    def set_motor_report_period_nowait(self, period_cmd: int)->None:
        # TODO: use Enum.
        # assert period_ms in {10,15,20,25,30,35,40,45,50,55,60}
        assert period_cmd in {ReportPeriodCmd.P_10MS, ReportPeriodCmd.P_15MS,
                             ReportPeriodCmd.P_20MS, ReportPeriodCmd.P_25MS,
                             ReportPeriodCmd.P_30MS, ReportPeriodCmd.P_35MS,
                             ReportPeriodCmd.P_40MS, ReportPeriodCmd.P_45MS,}
        # cmd = period_ms // 5 - 1  # 0 is 10ms, then add 1 for every 5ms.
        self._write_param_tx_helper('EPScan_time', [period_cmd] * len(self._motor_can_id_ordering))

    def read_motor_report_period_tx(self)->None:
        self._read_param_tx_helper('EPScan_time')

    def get_motor_report_period(self, timeout_sec: float)->npt.NDArray[np.float32|np.int32]:
        return self._get_param_value_helper('EPScan_time', timeout_sec)

    def _enable_motor_nowait(self, ids:Sequence[int])->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.motor_enable(motor_can_id=ids,
                                                                   host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def _disable_motor_nowait(self, ids:Sequence[int])->None:
        snd_msg:List[can.Message] = RSProtocolBuilder.motor_disable(motor_can_id=ids,
                                                                    host_can_id=self._host_can_id)
        self._tx_msg(snd_msg)

    def disable_motors(self, ids:Sequence[int]=None):
        if ids is not None:
            disable_id = tuple(self._set_of_motor_can_id & set(ids))
        else:
            disable_id = self._motor_can_id_ordering

        logger.info(f"disable motor id: {disable_id}")
        self._disable_motor_nowait(disable_id)

    def enable_motors(self, ids:Sequence[int]=None):
        if ids is not None:
            enable_id = tuple(self._set_of_motor_can_id & set(ids))
        else:
            enable_id = self._motor_can_id_ordering

        logger.info(f"enable motor id: {enable_id}")
        self._enable_motor_nowait(enable_id)

    def __del__(self):
        pass
        # in case the io_proc still alive
        # self.disable_motors(ids=None)

    # def read_model_number_tx(self, wait_sec: float)->None:
    #     snd_msg: List[can.Message] = RSProtocolBuilder.get_device_id(motor_can_id=self._motor_can_id,
    #                                                                 host_can_id=self._host_can_id)
    #     self._tx_msg(snd_msg)



if __name__ == '__main__':
    # import concurrent.futures
    import asyncio
    # import threading
    import atexit
    from toddlerbot.actuation.robstride_io_proc import RobStrideIOProc

    config_logging(root_logger_level=logging.INFO, root_handler_level=logging.NOTSET,
                   root_fmt='--- {levelname} - module:{module} - func:{funcName} ---> \n{message}',
                   root_date_fmt='%Y-%m-%d %H:%M:%S',
                   # log_file='/tmp/toddler/imitate_episode.log',
                   log_file=None,
                   module_logger_config={'robstride_io_proc': logging.DEBUG})

    def mock_cpu_bound_policy(ctrl: BaseController):
        time.sleep(2.0)
        print(f'---> start initialize motors')
        ctrl.initialize_motors()
        print(f'finish initialize motors <---')

        ret : int = 0
        # while True:
        for _ in range(5):
            time.sleep(1.)
            ret+=1
            print(f'---cpu bound --- {ret=:} ----')
            if ret > 0xffffff:
                ret = 0

    def run_io_bound_task_in_spawned_process(*, motor_can_id: Sequence[int],         # ids of a group of actuators.
                                                host_can_id: int,
                                                channel: str,
                                                baud_rate: RSBaudRate,
                                                event_conn: Connection,
                                                ctrl_msg_q: mp.Queue,  #[ControlMsg],
                                                motor_state_frame_q: mp.Queue, # [MotorStateFrame],
                                                motor_param_value_q: mp.Queue, # [SingleParamValue]
                                             ):

        _proc = RobStrideIOProc(motor_can_id=motor_can_id,
                                host_can_id=host_can_id,
                                channel=channel,
                                baud_rate=baud_rate,
                                event_conn_with_controller_proc=event_conn,
                                ctrl_msg_q=ctrl_msg_q,
                                motor_state_frame_q=motor_state_frame_q,
                                motor_param_value_q=motor_param_value_q)

        return asyncio.run(_proc.send_rcv_task())

    _cfg = RobStrideConfig(channel='can0',
                          baud_rate=RSBaudRate.BPS_1M,
                          run_mode=RSRunMode.PP_POSITION,
                          motor_report_period=RSReportPeriod.P_40MS,
                          host_can_id=0xfe,
                          motor_can_id=[0x7f],
                          pos_kp=None,
                          default_accel_PP_mode=None,
                          default_vel_PP_mode=None,
                          init_target_pos=np.asarray([0.],dtype=np.float32))

    # NOTE: mp.Queue is always preferable, causing it is a high-level API rather than sync-primitive.
    # mp.Queue using BoundedSemaphore to control the queue size. better than SimpleQueue which is unbounded.
    # and mp.Queue creates a uni-directional connection Pipe(duplex=False).
    # also better than mp.Pipe which has no management of queue size neither, just using OS PIPE to send/rcv.
    _motor_ctrl_q = mp.Queue(maxsize=100)
    _motor_state_frame_q = mp.Queue(maxsize=100)
    _motor_param_value_q = mp.Queue(maxsize=100)

    # duplex Pipe, used for exchange simple events between main proc and io proc.
    # NOTE: if want to exchange event among multi-processes, use mp.Queue.
    _io_proc_event_conn: Connection
    _main_proc_event_conn: Connection
    # pair of ends, used by each proc.
    _io_proc_event_conn, _main_proc_event_conn = mp.Pipe(duplex=True)

    controller = RobStrideController(config = _cfg,
                                     event_conn_with_io_proc=_main_proc_event_conn,
                                     motor_ctrl_q= _motor_ctrl_q,
                                     motor_state_frame_q= _motor_state_frame_q,
                                     motor_param_value_q= _motor_param_value_q)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as p_pool:
    #     fut:concurrent.futures.Future = p_pool.submit(run_io_bound_task_in_process_pool, controller)

    # NOTE: python `daemon` process mimic the behavour of thread, which will be terminated after the parent process
    # terminates, not the concept of Linux/Unix daemon services which will kept at backgroud even after the parent
    # process terminates.
    _io_proc = mp.Process(target=run_io_bound_task_in_spawned_process,
                         args=[],
                         kwargs=dict(motor_can_id=_cfg.motor_can_id,
                                     host_can_id=_cfg.host_can_id,
                                     channel=_cfg.channel,
                                     baud_rate=_cfg.baud_rate,
                                     event_conn=_io_proc_event_conn,
                                     ctrl_msg_q=_motor_ctrl_q,
                                     motor_state_frame_q=_motor_state_frame_q,
                                     motor_param_value_q=_motor_param_value_q),
                         name='asyncio_send_rcv_can_msg',
                         daemon=True)

    # io_thrd = threading.Thread(target=run_io_bound_task_in_spawned_process,args=[controller],daemon=True)

    # terminate gracefully.
    # def clean_io_process():
    #     print(f'clean_io_process(): terminate io_process gracefully.')
    #     while _io_proc.is_alive():
    #         print(f'_io_proc is still alive: {_io_proc.is_alive()}, terminate it')
    #         _io_proc.terminate()
    #         time.sleep(0.5)
    #     # fut.cancel()
    #     print(f'_io_proc is alive: {_io_proc.is_alive()}')
    #     # controller.close_motors()
    #     time.sleep(1.)

    # def clean__io_process():
    #     print(f'clean__io_process(): terminate _io_process gracefully.')
    #     while io_thrd.is_alive():
    #         print(f'_io_proc is still alive: {io_thrd.is_alive()}, terminate it')
    #         io_thrd.terminate()
    #         time.sleep(0.5)
    #     # fut.cancel()
    #     print(f'_io_proc is alive: {io_thrd.is_alive()}')
    #     print(f'close motors:')
    #     controller.close_motors()
    #     time.sleep(0.5)

    def exit_handler():
        print(f'###### \n\ncalled from exit_handler of main proces/main thread: ---> \n\n ######')
        clean_children_proc()

    atexit.register(exit_handler)

    def clean_children_proc():
        print(f'##### clean IO proc ---> ##### ')
        for _p in  mp.active_children():
            print(f'#####  active child process name:{_p.name} #####')

            if _p.name == 'asyncio_send_rcv_can_msg':
                print(f'##### IO proc still alive, start to clean ip proc ---> #####')
                _main_proc_event_conn.send(RSIOEvent.ReqIODisconnect)
                last_event = _main_proc_event_conn.recv()
                if last_event is RSIOEvent.DoneIODisconnect:
                    logger.warning(f'##### recv RSIOEvent.DoneIODisconnect from io proc, we can finish main process. ##### ' )
                else:
                    raise ValueError(f' ### recv unexpected event from io proc:{last_event} ###')

            time.sleep(2.)
            while _p.is_alive():
                print(f'proc:{_p.name} is still alive: {_p.is_alive()}, terminate it')
                _p.terminate()
                time.sleep(0.5)

    try:
        print(f'start _io_process.')
        _io_proc.start()
        # _io_proc.join()
        print(f'_io_proc is alive: {_io_proc.is_alive()}')
        print(f'start mock cpu bound policy.')
        # TODO: naive solution to wait for the io task running. maybe using connection?
        while not _io_proc.is_alive():
            time.sleep(0.5)
        mock_cpu_bound_policy(controller)

    # try:
    #     print(f'start _io_process.')
    #     io_thrd.start()
    #     # _io_proc.join()
    #     print(f'_io_proc is alive: {io_thrd.is_alive()}')
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
        # clean_io_process()

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
