"""
run policy(cpu bound) and RS IO Proc(io bound) through multiprocessing. for real world.
"""

import asyncio
import atexit
import logging
import time
from typing import Sequence
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection

from toddlerbot.actuation._module_logger import logger
from toddlerbot.utils.config_logging import config_logging
from toddlerbot.actuation.base_controller import JointState
from toddlerbot.actuation.robstride_io_proc import (RobStrideIOProc, RSIOEvent,
                                                    RSBaudRate, RSRunMode, RSReportPeriod )
from toddlerbot.actuation.robstride_control import ( RobStrideConfig,
                                                     RobStrideController )

from toddlerbot.actuation.robstride_sdk.rs_bandwidth_test import MotorBandwidthTester, Action

MOTOR_CAN_ID = 0x7f
HOST_CAN_ID =0xfe

def _args_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='mp run a policy.')
    # TODO: confusing.  we can separate them into two fields:  --policy xxx  --fixed true/false.
    # parser.add_argument(
    #     "--policy",
    #     type=str,
    #     default="stand",
    #     help="The name of the task.",
    #     choices=_get_policy_names_v2(),
    # )
    return parser.parse_args()

# define IO Proc target func:
def _run_io_bound_task_in_spawned_process(*, motor_can_id: Sequence[int],  # ids of a group of actuators.
                                         host_can_id: int,
                                         channel: str,
                                         baud_rate: RSBaudRate,
                                         event_conn: Connection,
                                         ctrl_msg_q: mp.Queue,  # [ControlMsg],
                                         motor_state_frame_q: mp.Queue,  # [MotorStateFrame],
                                         motor_param_value_q: mp.Queue,  # [SingleParamValue]
                                         )->None:
    _proc = RobStrideIOProc(motor_can_id=motor_can_id,
                            host_can_id=host_can_id,
                            channel=channel,
                            baud_rate=baud_rate,
                            event_conn_with_controller_proc=event_conn,
                            ctrl_msg_q=ctrl_msg_q,
                            motor_state_frame_q=motor_state_frame_q,
                            motor_param_value_q=motor_param_value_q)
    return asyncio.run(_proc.send_rcv_task())

def _cpu_bound_policy(ctrl: RobStrideController):
    bd_tester = MotorBandwidthTester(sample_rate=25)  # 40ms motor report interval.

    print(f'---> start initialize motors')
    ctrl.initialize_motors()
    print(f'finish initialize motors <---')

    bd_tester.pre_process()

    # TODO: use barrier to sync?
    ctrl.toggle_periodic_report_nowait(enable=True)

    action = Action(value=None,last=False)
    # use `last` to send last obs of last action to bd_tester.
    while not action.last:
        # block waiting for periodic motor report.
        obs:JointState = ctrl.get_motor_state(None)[MOTOR_CAN_ID]
        logger.debug(f' === get joint state: {obs}  ===')
        action = bd_tester.step(obs)

        if action.value is not None:
            ctrl.set_pos([action.value])

    bd_tester.post_process()


def _main(args: argparse.Namespace):

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

    def _clean_children_proc():
        print(f'##### clean IO proc ---> ##### ')
        for _p in mp.active_children():
            print(f'#####  active child process name:{_p.name} #####')

            if _p.name == 'asyncio_send_rcv_can_msg':
                print(f'##### IO proc still alive, start to clean ip proc ---> #####')
                _main_proc_event_conn.send(RSIOEvent.ReqIODisconnect)
                last_event = _main_proc_event_conn.recv()
                if last_event is RSIOEvent.DoneIODisconnect:
                    logger.warning(
                        f'##### recv RSIOEvent.DoneIODisconnect from io proc, we can finish main process. ##### ')
                else:
                    raise ValueError(f' ### recv unexpected event from io proc:{last_event} ###')

            time.sleep(2.)
            while _p.is_alive():
                print(f'proc:{_p.name} is still alive: {_p.is_alive()}, terminate it')
                _p.terminate()
                time.sleep(0.5)

    def _exit_handler():
        print(f'###### \n\ncalled from exit_handler of main proces/main thread: ---> \n\n ######')
        _clean_children_proc()

    """
     NOTE: `atexit` only works in main process/main thread.
    """
    atexit.register(_exit_handler)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as p_pool:
    #     fut:concurrent.futures.Future = p_pool.submit(run_io_bound_task_in_process_pool, controller)

    _rs_cfg = RobStrideConfig(channel='can0',
                              baud_rate=RSBaudRate.BPS_1M,
                              run_mode=RSRunMode.PP_POSITION,
                              motor_report_period=RSReportPeriod.P_40MS,
                              host_can_id=HOST_CAN_ID,
                              motor_can_id=[MOTOR_CAN_ID],
                              pos_kp=[30],
                              default_accel_PP_mode=[90],
                              default_vel_PP_mode=[40],
                              init_target_pos=np.asarray([0.], dtype=np.float32))

    # NOTE: python `daemon` process mimic the behaviour of thread, which will be terminated after the parent process
    # terminates, not the concept of Linux/Unix daemon services which will kept at background even after the parent
    # process terminates.
    _io_proc = mp.Process(target=_run_io_bound_task_in_spawned_process,
                          args=[],
                          kwargs=dict(motor_can_id=_rs_cfg.motor_can_id,
                                      host_can_id=_rs_cfg.host_can_id,
                                      channel=_rs_cfg.channel,
                                      baud_rate=_rs_cfg.baud_rate,
                                      event_conn=_io_proc_event_conn,
                                      ctrl_msg_q=_motor_ctrl_q,
                                      motor_state_frame_q=_motor_state_frame_q,
                                      motor_param_value_q=_motor_param_value_q),
                          name='asyncio_send_rcv_can_msg',
                          daemon=True)

    controller = RobStrideController(config=_rs_cfg,
                                     event_conn_with_io_proc=_main_proc_event_conn,
                                     motor_ctrl_q=_motor_ctrl_q,
                                     motor_state_frame_q=_motor_state_frame_q,
                                     motor_param_value_q=_motor_param_value_q)
    try:
        logger.warning(f'start _io_process.')
        _io_proc.start()
        # _io_proc.join()
        logger.warning(f'_io_proc is alive: {_io_proc.is_alive()}')
        logging.warning(f'start mock cpu bound policy.')
        # TODO: naive solution to wait for the io task running. maybe using connection?
        while not _io_proc.is_alive():
            time.sleep(0.5)

        logger.warning(f'start cpu bound process.')
        _cpu_bound_policy(controller)

    except Exception as error:
        logger.error(f'--- exception in main process: {error=:} {type(error)=:}')
        time.sleep(0.5)
        raise error

    finally:
        # normal finish.
        logger.warning(f'finally: clean up in finally--->')
        time.sleep(0.5)


if __name__ == '__main__':

    _parsed_args = _args_parsing()
    # TODO: move into yaml config.
    config_logging(root_logger_level=logging.INFO, root_handler_level=logging.NOTSET,
                   root_fmt='--- {levelname} - module:{module} - func:{funcName} ---> \n{message}',
                   root_date_fmt='%Y-%m-%d %H:%M:%S',
                   # log_file='/tmp/toddler/imitate_episode.log',
                   log_file=None,
                   module_logger_config={'robstride_io_proc': logging.WARNING,
                                         'main': logging.INFO})
    # use root logger for __main__.
    logger = logging.getLogger('root')
    logger.info('parsed args --->\n{}'.format('\n'.join(
        f'{arg_name}={arg_value}' for arg_name, arg_value in
        sorted(_parsed_args.__dict__.items(), key=lambda k_v_pair: k_v_pair[0]))))

    _main(_parsed_args)

