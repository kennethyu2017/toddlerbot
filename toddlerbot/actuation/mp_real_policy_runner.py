"""
run policy(cpu bound) and RS IO Proc(io bound) through multiprocessing. for real world.
"""

import asyncio
import atexit
import logging
import time
from typing import Sequence, List
from collections import OrderedDict
import argparse
import time as timelib
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import numpy.typing as npt
from pathlib import Path
import pickle
import multiprocessing as mp
from multiprocessing.connection import Connection
from tqdm import tqdm

from toddlerbot.actuation._module_logger import logger
from toddlerbot.utils.config_logging import config_logging
from toddlerbot.actuation.base_controller import JointState
from toddlerbot.actuation.robstride_io_proc import (RobStrideIOProc, RSIOEvent,
                                                    RSBaudRate, RSRunMode, RSReportPeriod )
from toddlerbot.actuation.robstride_control import ( RobStrideConfig,
                                                     RobStrideController )

# from toddlerbot.actuation.robstride_sdk.rs_bandwidth_test import MotorBandwidthTester, Action
from toddlerbot.policies import (Action, StepRecord,RUN_POLICY_LOG_FOLDER_FMT,
                                 RUN_STEP_RECORD_PICKLE_FILE,RUN_EPISODE_MOTOR_KP_PICKLE_FILE)
from toddlerbot.policies.run_policy import _plot_run_log
from toddlerbot.actuation.robstride_sdk.mp_rs_sysID import MpSysIDPolicy,MockRobot, MotorKpSetter
from toddlerbot.sim import Obs

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

# def _cpu_bound_motor_bd_width_policy(ctrl: RobStrideController):
#     bd_tester = MotorBandwidthTester(sample_rate=25)  # 40ms motor report interval.
#
#     print(f'---> start initialize motors')
#     ctrl.initialize_motors()
#     print(f'finish initialize motors <---')
#
#     bd_tester.pre_process()
#
#     # TODO: use barrier to sync?
#     ctrl.toggle_periodic_report_nowait(enable=True)
#
#     action = Action(value=None,last=False)
#     # use `last` to send last obs of last action to bd_tester.
#     while not action.last:
#         # block waiting for periodic motor report.
#         obs:JointState = ctrl.get_motor_state(None)[MOTOR_CAN_ID]
#         logger.debug(f' === get joint state: {obs}  ===')
#         action = bd_tester.step(obs)
#
#         if action.value is not None:
#             ctrl.set_pos([action.value])
#
#     bd_tester.post_process()


def _save_run_log(step_record_list: List[StepRecord], pickle_file: Path):
    # log_dir = exp_folder / 'step_record'
    if not pickle_file.parent.exists():
        pickle_file.parent.mkdir(parents=True)

    # with open(log_dir / 'step_record_list.pkl', 'wb') as _f:
    with open(pickle_file, 'wb') as _f:
        pickle.dump(step_record_list, _f)


def _save_policy_log(*, policy: MpSysIDPolicy,
                     log_dir: Path,
                     step_record_list: List[StepRecord]):
    # log_dir = exp_folder / policy.name
    if not log_dir.exists():
        log_dir.mkdir()

    # with open(log_dir/'episode_motor_kp.pkl', "wb") as _f:
    with open(log_dir / RUN_EPISODE_MOTOR_KP_PICKLE_FILE,  # .format(policy_name=policy.name),
              "wb") as _f:
        pickle.dump(policy.episode_info, _f)


def _cpu_bound_sysID_policy(rs_ctrl: RobStrideController):

    sysID_policy = MpSysIDPolicy(init_motor_pos=np.asarray([0.],dtype=np.float32),
                                 jnt_cfg_limit=(-np.pi/2, np.pi/2),
                                 control_dt_sec=0.04)  # 40ms motor report interval.

    print(f'---> start initialize motors')
    rs_ctrl.initialize_motors()
    print(f'finish initialize motors <---')

    step_record_list: List[StepRecord] = []
    _step_count: int = 0
    # update tqdm every 1 sec.
    p_bar_steps: int = max(1, int(1 / sysID_policy.control_dt_sec))

    # for sysID only.
    # _cur_ep_idx :int = -1
    motor_kp_setter = MotorKpSetter()

    # TODO: for tqdm,  if total is float('inf'), Infinite iterations,
    #  behave same as `total-unknown`: can not show progress bar.
    # not use tqdm for n_steps_total is inf?
    sysID_policy.pre_process()

    with  (tqdm(total=sysID_policy.n_steps_total, desc="Running the policy",
               colour='CYAN', unit='step', unit_scale=True) as p_bar):

        run_start_time = timelib.time()
        run_start_perf_cnt = timelib.perf_counter()

        try:
            # TODO: use barrier to sync?
            rs_ctrl.toggle_periodic_report_nowait(enable=True)
            # action = Action(value=None, last=False)
            # while not action.last:
            while _step_count < sysID_policy.n_steps_total:
                # TODO: temply try.
                _loop_start_ns: int = timelib.perf_counter_ns()

                _record = StepRecord()
                _record.time_pnt.step_start = timelib.time()

                # Get the latest state from the queue
                jnt_state: JointState = rs_ctrl.get_motor_state(None)[MOTOR_CAN_ID]
                logger.debug(f' === get joint state: {jnt_state}  ===')

                # change to epoch time.
                jnt_state.time -= run_start_time

                _record.time_pnt.recv_obs = timelib.perf_counter() - run_start_perf_cnt

                # for sysID policy to change motor kp if kp changed.
                motor_kp_setter.set_kp(policy=sysID_policy,
                                       ctrl=rs_ctrl,
                                       step_count=_step_count,
                                       obs_time=jnt_state.time)

                control_inputs, motor_target_arr = sysID_policy.step(jnt_state)
                _record.time_pnt.inference = timelib.perf_counter() - run_start_perf_cnt

                if _step_count % 50 == 1:
                    # NOTE: set/get value should be normalized by feite_controller.init_pos
                    logger.info(f'prev act:{step_record_list[-1].motor_act}, {jnt_state.pos=:}, {motor_target_arr=:}')

                # env.set_motor_target(motor_angle_dict)
                rs_ctrl.set_pos(motor_target_arr)
                _record.time_pnt.set_action = timelib.perf_counter() - run_start_perf_cnt
                _record.time_pnt.sim_step = timelib.perf_counter() - run_start_perf_cnt

                _record.obs = Obs(time=jnt_state.time,
                                  motor_pos= np.asarray([jnt_state.pos], dtype=np.float32),
                                  motor_vel= np.asarray([jnt_state.vel], dtype=np.float32),
                                  motor_tor= np.asarray([jnt_state.tor], dtype=np.float32))

                _record.ctrl_input = deepcopy(control_inputs)
                _record.motor_act = deepcopy(motor_target_arr)

                _step_count += 1

                # update tqdm every 1 sec (time measured in policy.control_dt).
                if _step_count % p_bar_steps == 0:
                    p_bar.update(p_bar_steps)

                _record.time_pnt.step_end = timelib.perf_counter() - run_start_perf_cnt
                step_record_list.append(_record)

        except KeyboardInterrupt:
            # only catch Keyboard Interrupt as normal exit from while loop,
            # and save running logs in and after `finally` block.
            logger.warning("KeyboardInterrupt received. exit while loop, and save running logs.")

        except Exception as err:
            # other exceptions, like IOError, re-raise the exception to outer `try.. ex...fi..`.
            # without saving running logs.
            # NOTE: the `finally` block will be executed before re-raise to outer `try` block.
            logger.error(f'Unexpected error occurred: {err=:}, {type(err)=:}. re-raise to outer handler.')
            raise

        finally:
            # p_bar.close()
            logger.info(f'exit from run while loop, final step_count: {_step_count},'
                        f' step record count: {len(step_record_list)}')

            # TODO: save recording file every n steps n seconds. ... not at the end of while loop.....
            # exp_name = f"{robot.name}_{policy.name}_{env.env_name}"
            # exp_folder = Path('run_policy_log') / f'{exp_name}_{cur_time}'
            # 'run_policy_log/{robot_name}_{policy_name}_{env_name}_{cur_time}'
            cur_time = timelib.strftime("%Y%m%d_%H%M%S")
            exp_folder = Path(RUN_POLICY_LOG_FOLDER_FMT.format(robot_name='RS02',
                                                               policy_name='MpSysID',
                                                               env_name='sysID',
                                                               cur_time=cur_time))
            if not exp_folder.exists():
                exp_folder.mkdir(parents=True, exist_ok=True)

            # Using context mgr to close env, not use close() standalone.
            # close() also set torque off for all connected motors.
            # env.close()

            # ----  at end of `finally` execution, if there is un-handled Exp, will raise to outer `try` block; else,
            # execution continues the following code.

    # ---- save logs only when: 1. finish while loop; 2. KeyboardInterrupt. ----

    # TODO: write log data every n steps..n seconds.. not at the end of while loop.....
    _save_run_log(step_record_list, exp_folder / RUN_STEP_RECORD_PICKLE_FILE)
    _save_policy_log(policy=sysID_policy,
                     log_dir=exp_folder,  # / policy.name,
                     step_record_list=step_record_list)

    logger.info("--- Plot policy run logg --->")
    mock_rbt = MockRobot('RS_sysID')

    _plot_run_log(
        mock_rbt,
        sysID_policy,
        step_record_list,
        exp_folder / 'plot'
    )

    sysID_policy.post_process()


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
                              default_accel_PP_mode=[190],
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
        # _cpu_bound_policy(controller)
        _cpu_bound_sysID_policy(controller)

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

