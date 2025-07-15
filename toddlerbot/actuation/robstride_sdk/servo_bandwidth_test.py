import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from typing import Tuple

from toddlerbot.actuation.feite_servo.scservo_sdk import SMS_STS_DEFAULT_BAUD_RATE
from toddlerbot.actuation.feite_control import (FeiteController, FeiteConfig)
from toddlerbot.sim.real_world import _DEFAULT_FEITE_VEL,_DEFAULT_FEITE_ACCEL,_DEFAULT_FEITE_TORQUE_LIMIT_PERCENTAGE

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define constants.
URT_1_DEV_NAME : str = r'/dev/ttyUSB0'
MOTOR_ID: int = 0

DEFAULT_KP: int = 32
DEFAULT_KD: int = 2

SIGNAL_AMPLITUDE: float = 0.5 * np.pi/2.
POS_LIMIT: float = 0.97 * np.pi / 2.

START_FREQ = 0.5
END_FREQ = 7 #20
FREQ_SAMPLES = 10

SAMPLE_RATE :int = 400 # 1000
DWELL_TIME= 3 # sec

def _init_feite_actuators()->FeiteController:
    feite_ids = [0]
    control_mode = ['position']
    kP = [DEFAULT_KP]
    kI = [0]
    kD = [DEFAULT_KD]
    init_pos = np.asarray([3.1415926],dtype=np.float32)
    default_torque_limit=np.asarray([_DEFAULT_FEITE_TORQUE_LIMIT_PERCENTAGE],dtype=np.float32)
    default_accel=np.asarray([_DEFAULT_FEITE_ACCEL], dtype=np.float32)
    default_vel=np.asarray([_DEFAULT_FEITE_VEL], dtype=np.float32)
    return_delay_us = 250


    feite_config = FeiteConfig(
        port=URT_1_DEV_NAME,
        baudrate=SMS_STS_DEFAULT_BAUD_RATE,
        control_mode=control_mode,
        kP=kP,
        kI=kI,
        kD=kD,
        default_torque_limit=default_torque_limit ,
        default_accel=default_accel,
        default_vel=default_vel,
        init_goal_pos=init_pos,
        return_delay_us=return_delay_us
    )

    controller: FeiteController|None = None

    try:
        controller = FeiteController(feite_config, feite_ids)

    except KeyboardInterrupt:
        if controller is not None:
            controller.close_motors()
        logger.error(f'keyboard interrupt, exit...')
        exit(1)

    except Exception as exc:
        logger.error(f'type of exc: {type(exc)}')
        if controller is not None:
            controller.close_motors()
        raise exc

    else:
        return controller


class ServoBandwidthTester:
    def __init__(self, *,
                 # com_port: str, baudrate: int = 115200,
                 pos_limit: float = POS_LIMIT,
                 sample_rate: int):
        """
        初始化伺服电机带宽测试器
        
        参数:
            com_port: 串口通信端口
            baudrate: 波特率
            voltage_limit: 最大输出电压限制(V)
            sample_rate: 采样率(Hz)
        """
        # self.com_port = com_port
        # self.baudrate = baudrate
        self.pos_limit = pos_limit
        self.sample_rate:int = sample_rate
        self.freq_response_data = []

        self.controller = _init_feite_actuators()
        
    # def connect_motor(self) -> bool:
    #     """连接到伺服电机"""
    #     try:
    #         # 这里应替换为实际的串口或网络连接代码
    #         # 例如: self.serial = serial.Serial(self.com_port, self.baudrate, timeout=1)
    #         logger.info(f"成功连接到电机，端口: {self.com_port}, 波特率: {self.baudrate}")
    #         self.motor_connected = True
    #         return True
    #     except Exception as e:
    #         logger.error(f"连接电机失败: {str(e)}")
    #         return False
    
    def disconnect_motor(self) -> None:
        """断开与电机的连接"""
        if self.controller is not None:
            # 关闭输出，确保电机停止
            # self._set_motor_pos(0.0)
            # 这里应替换为实际的关闭连接代码
            # 例如: self.serial.close()
            # self.controller.close_motors()
            # self.controller = None

            logger.info("已断开与电机的连接")
    
    def _set_motor_pos(self, pos: float) -> bool:
        """
        设置电机电压
        
        参数:
            voltage: 目标电压(V)
        
        返回:
            设置是否成功
        """
        # 电压安全限制
        # voltage = max(-self.pos_limit, min(voltage, self.pos_limit))

        assert abs(pos) < self.pos_limit
        
        try:
            self.controller.set_pos([pos])

            # logger.debug(f"设置pos: {pos:.2f} rad")
            return True

        except Exception as e:
            logger.error(f"设置pos失败: {str(e)} {type(e)}")
            raise e
    
    def _read_motor_response(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取电机响应数据
        
        参数:
            duration: 采样持续时间(秒)
        
        返回:
            时间数组和位置反馈数组
        """
        samples = int(duration * self.sample_rate)
        # time_data = np.linspace(0, duration, samples)
        position_data = np.zeros(samples)
        time_data = np.zeros_like(position_data)
        sample_interval = 1.0 / self.sample_rate

        # logger.info(f'{position_data.shape=:}')
        
        try:
            for _i in range(samples):

                _cnt1 = time.perf_counter_ns()
                state_dict = self.controller.get_motor_state()
                print(f'get obs ms: {((time.perf_counter_ns() - _cnt1) / 1_000_000):.2f}')

                position_data[_i] = state_dict[MOTOR_ID].pos
                time_data[_i] = state_dict[MOTOR_ID].time
                # time.sleep(sample_interval)
            
            # 模拟读取数据，实际应用中应替换为真实的传感器数据
            # position_data = np.random.normal(0, 0.1, samples)
            
            return time_data, position_data
        except Exception as e:
            logger.error(f"读取响应数据失败: {str(e)} {type(e)}")
            raise e
    
    def run_frequency_sweep(self,*,
                            start_freq: float = 0.5, end_freq: float = 200.0,
                            num_freq: int = 50,
                            dwell_time: float = 2.0,
                            amplitude: float,
                            ) -> bool:
        """
        执行频率扫描测试
        
        参数:
            start_freq: 起始频率(Hz)
            end_freq: 终止频率(Hz)
            num_freqs: 频率点数量
            amplitude: 输入信号幅值(V)
            dwell_time: 每个频率点的持续时间(秒)
        
        返回:
            测试是否成功
        """
        if self.controller is None:
            logger.error("电机未连接...")
            return False
        
        # 生成对数分布的测试频率
        test_frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), num_freq)
        self.freq_response_data = []
        
        logger.info(f"开始频率扫描测试: {start_freq}Hz 到 {end_freq}Hz, {num_freq} 个频率点...")

        time.sleep(1)
        
        try:
            # 先让电机稳定在零位
            # self._set_motor_voltage(0.0)

            # normalized position:
            self.controller.set_pos([0.0])
            time.sleep(1.0)
            
            for freq in test_frequencies:
                logger.info(f"测试频率: {freq:.2f}Hz")
                
                # 生成当前频率的正弦信号

                signal_periods:int = int(np.ceil(dwell_time * freq))
                rad = np.linspace(0, signal_periods*2*np.pi, int(signal_periods/freq  * self.sample_rate))
                sine_wave = amplitude * np.sin(rad)

                # t = np.linspace(0, dwell_time, int(dwell_time * self.sample_rate))
                # sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
                logger.info(f'{sine_wave.shape=:} {max(sine_wave)=:} {min(sine_wave)=:} '
                            f'{sine_wave[0]=:} {sine_wave[-1]=:}')

                loop_interval_ns :int = round(1_000_000_000/self.sample_rate)

                # 发送正弦信号并记录响应
                response_times = []
                response_positions = []

                for _pos in sine_wave:

                    loop_start_ns = time.perf_counter_ns()

                    # assert abs(_pos) < POS_LIMIT
                    # logger.info(f'--- {_pos=:} ---')

                    self._set_motor_pos(_pos)

                    # 等待电机响应
                    # time.sleep(0.001)
                    until_next_step_sec = (loop_start_ns
                                          + loop_interval_ns
                                          - time.perf_counter_ns()) /1_000_000_000

                    logger.debug(f"until_next_step_ns: {until_next_step_sec:.4f} sec")

                    if until_next_step_sec > 0:
                        # logger.debug(f'+++++ sleep for {until_next_step_sec:.4f} sec ')
                        time.sleep(until_next_step_sec)

                    # 读取当前位置 (实际应用中可能需要更高效的读取方式)
                    time_seq, pos_seq = self._read_motor_response(duration=1./self.sample_rate)  # 0.01)

                    response_times.extend(time_seq)
                    response_positions.extend(pos_seq)


                logger.info(f'{len(response_positions)=:}')

                time.sleep(2)
                # 分析频率响应
                input_fft = np.fft.rfft(sine_wave)
                output_fft = np.fft.rfft(response_positions)
                
                # 找到测试频率对应的FFT索引
                freq_idx = int(freq * dwell_time)
                
                # 计算幅值比和相位差
                input_mag = np.abs(input_fft[freq_idx])
                output_mag = np.abs(output_fft[freq_idx])
                phase_diff = np.angle(output_fft[freq_idx] / input_fft[freq_idx], deg=True)
                
                # 计算增益 (dB)
                gain_db = 20 * np.log10(output_mag / input_mag) if input_mag > 0 else -np.inf
                
                self.freq_response_data.append({
                    'frequency': freq,
                    'gain_db': gain_db,
                    'phase_deg': phase_diff,
                    'input_mag': input_mag,
                    'output_mag': output_mag
                })
                
                # 短暂暂停，避免频率切换过快
                time.sleep(0.5)
            
            logger.info("频率扫描测试完成")
            return True
            
        except Exception as e:
            logger.error(f"频率扫描测试失败: {str(e)}")
            raise e
    
    def calculate_bandwidth(self, low_freq_gain: float = None) -> float:
        """
        计算闭环带宽
        
        参数:
            low_freq_gain: 低频增益参考值，若为None则使用第一个频率点的增益
        
        返回:
            闭环带宽(Hz)
        """
        if not self.freq_response_data:
            logger.error("没有频率响应数据，请先运行频率扫描测试")
            return 0.0
        
        # 确定参考增益 (通常是低频增益)
        if low_freq_gain is None:
            # 使用前5个频率点的平均增益作为参考
            # low_freq_points = min(5, len(self.freq_response_data))
            # use first point.
            low_freq_points = 1
            low_freq_gain = np.mean([data['gain_db'] for data in self.freq_response_data[:low_freq_points]])
        
        # 计算-3dB点
        target_gain = low_freq_gain - 3.0

        logger.info(f'{target_gain=:}')
        
        # 找到第一个低于目标增益的频率点
        bandwidth = 0.0
        for i, data in enumerate(self.freq_response_data):
            if data['gain_db'] < target_gain:
                if i > 0:
                    # 线性插值以获得更准确的带宽估计
                    prev_data = self.freq_response_data[i-1]
                    x0, y0 = prev_data['frequency'], prev_data['gain_db']
                    x1, y1 = data['frequency'], data['gain_db']
                    
                    # 插值公式: x = x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)
                    bandwidth = x0 + (target_gain - y0) * (x1 - x0) / (y1 - y0)
                else:
                    bandwidth = data['frequency']
                break
        
        # 如果没有找到低于-3dB的点，返回最高测试频率
        if bandwidth == 0.0 and self.freq_response_data:
            bandwidth = self.freq_response_data[-1]['frequency']
        
        logger.info(f"计算得到闭环带宽: {bandwidth:.2f}Hz")
        return bandwidth
    
    def plot_frequency_response(self, save_path: Path = None) -> None:
        """
        绘制频率响应曲线
        
        参数:
            save_path: 图像保存路径，若为None则显示图像
        """
        if not self.freq_response_data:
            logger.error("没有频率响应数据，请先运行频率扫描测试")
            return
        
        frequencies = [data['frequency'] for data in self.freq_response_data]
        gains_db = [data['gain_db'] for data in self.freq_response_data]
        phases_deg = [data['phase_deg'] for data in self.freq_response_data]
        
        # 计算带宽
        bandwidth = self.calculate_bandwidth()
        
        plt.figure(figsize=(12, 8))
        
        # 绘制幅频特性曲线
        plt.subplot(2, 1, 1)
        plt.plot(frequencies, gains_db, 'b-',linewidth=2)
        # plt.semilogx(frequencies, gains_db, 'b-', linewidth=2)
        # plt.plot(frequencies, gains_db, 'b-', linewidth=2)

        plt.axhline(y=-3, color='r', linestyle='--', label='-3dB')
        plt.axvline(x=bandwidth, color='g', linestyle='--', label=f'Bandwidth = {bandwidth:.2f}Hz')
        plt.xlabel('Freq (hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Servo closed-loop freq actuation')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # 绘制相频特性曲线
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, phases_deg, 'b-', linewidth=2)
        # plt.semilogx(frequencies, phases_deg, 'b-', linewidth=2)

        plt.axhline(y=-90, color='r', linestyle='--', label='-90°')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Phase (Degree)')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()

        save_file=  save_path / "actuation_{}_kp_{}_kd_{}.svg".format( time.strftime('%Y%m%d_%H%M%S'),
                                                                       DEFAULT_KP, DEFAULT_KD)

        if save_path:
            plt.savefig(save_file,bbox_inches='tight')
            # plt.savefig(save_path.resolve().__str__(), dpi=300, bbox_inches='tight')
            # plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logger.info(f"频率响应曲线已保存至: {save_path.resolve()}")
        else:
            plt.show()

def main():
    """主函数，演示如何使用伺服带宽测试器"""
    tester = ServoBandwidthTester(pos_limit=POS_LIMIT, sample_rate=SAMPLE_RATE)
    
    try:
        # 连接电机
        if tester.controller is not None:
            # 运行频率扫描测试
            if tester.run_frequency_sweep(start_freq=START_FREQ,
                                          end_freq=END_FREQ,
                                          num_freq=FREQ_SAMPLES,
                                          amplitude=SIGNAL_AMPLITUDE,   #2.0
                                          dwell_time=DWELL_TIME):

                plot_dir = Path('./servo_bandwidth_test')
                if not plot_dir.exists():
                    plot_dir.mkdir()

                # 绘制频率响应曲线
                tester.plot_frequency_response(save_path=plot_dir)
                
                # 计算并打印带宽
                bandwidth = tester.calculate_bandwidth()
                logger.warning(f"============ 伺服电机闭环带宽: {bandwidth:.2f} Hz ==================")
    
    finally:
        # 确保断开连接
        tester.disconnect_motor()
        time.sleep(1.)

if __name__ == "__main__":
    main()    