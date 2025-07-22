import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from typing import Tuple,List, Deque, NamedTuple, Dict, Any
from collections import deque
from dataclasses import dataclass, field

from ..base_controller import JointState
from ...visualization import *

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define constants.
DEFAULT_KP: int = 30
DEFAULT_KD: int = 2

SIGNAL_AMPLITUDE: float = 0.5 * np.pi/2.
POS_LIMIT: float = 0.97 * np.pi / 2.

START_FREQ = 0.5
END_FREQ = 7 #20
FREQ_SAMPLES = 10  # 3

# SAMPLE_RATE :int =50 #25 # 40ms per step.
DWELL_TIME= 3 # sec  每个频率点的持续时间(秒)

@dataclass(init=True)
class StepRecord:
    freq: float=0.
    action_seq: List[float] = field(default_factory=list)
    time_seq: List[float] = field(default_factory=list)
    pos_seq: List[float] = field(default_factory=list)

class Action(NamedTuple):
    value: float|None = None
    last: bool = False


class MotorBandwidthTester:
    def __init__(self, *,
                 sample_rate: int,
                 pos_limit: float = POS_LIMIT,
                 start_freq: float = START_FREQ,
                 end_freq: float = END_FREQ,
                 freq_samples: int = FREQ_SAMPLES,
                 dwell_time: float = DWELL_TIME,
                 amplitude: float= SIGNAL_AMPLITUDE,
                 ):
        self.pos_limit = pos_limit
        self.sample_rate:int = sample_rate
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.freq_samples = freq_samples
        self.dwell_time = dwell_time
        self.amplitude = amplitude


        # (freq, action_seq.)
        self.freq_action_seq :Deque[Tuple[float, Deque[float]]] = deque()
        self._gen_freq_action_seq()

        # init idx.
        # self._curr_freq, self._curr_action_seq = self.freq_action_seq.popleft()
        self._curr_action_seq: Deque[float]| None = None

        # 发送正弦信号并记录响应
        self.freq_response_record: List[StepRecord] = []
        self._curr_step_record = StepRecord()
        self.freq_response_record.append(self._curr_step_record)

        self.analyze_result: List[Dict[str, float | npt.NDArray[np.float32]]] = []
        self._sleep_cnt: int = 0

        self._first_obs_time: float |None = None

    def _gen_freq_action_seq(self):
        # 生成对数分布的测试频率
        log_freq = np.logspace(np.log10(self.start_freq), np.log10(self.end_freq), self.freq_samples)


        logger.info(f" gen freq action seq : {self.start_freq}Hz 到 {self.end_freq}Hz, {self.freq_samples} 个频率点...")

        for freq in log_freq:
            logger.info(f"测试频率: {freq:.2f}Hz")

            # 生成当前频率的正弦信号

            signal_periods: int = int(np.ceil(self.dwell_time * freq))
            rad = np.linspace(0, signal_periods * 2 * np.pi, int(signal_periods / freq * self.sample_rate))
            sine_wave = self.amplitude * np.sin(rad)

            # t = np.linspace(0, dwell_time, int(dwell_time * self.sample_rate))
            # sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
            logger.info(f'{sine_wave.shape=:} {max(sine_wave)=:} {min(sine_wave)=:} '
                        f'{sine_wave[0]=:} {sine_wave[-1]=:}')

            self.freq_action_seq.append((freq, deque(sine_wave)))


    def step(self, obs:JointState)->Action|None:
        """
        None means pause this frame.
        """
        ret: Action|None = None

        if self._first_obs_time is None:
            self._first_obs_time = obs.time

        if self._curr_action_seq is None:
            self._curr_step_record.freq, self._curr_action_seq = self.freq_action_seq.popleft()
            ret = Action(value=self._curr_action_seq.popleft())

        elif len(self._curr_action_seq) == 0:
            if len(self.freq_action_seq) == 0:
                logger.warning(f'=== finish pop all the freq action seq. ===')
                ret = Action(value=None,last=True)
            else:
                # before start new freq action seq, sleep 3 sec.
                if self._sleep_cnt < self.sample_rate * 3:
                    self._sleep_cnt+=1
                    ret = Action(value=None)
                else:
                    self._sleep_cnt = 0
                    # TODO: return None to sleep for 2sec?
                    self._curr_step_record=StepRecord()
                    self._curr_step_record.freq, self._curr_action_seq = self.freq_action_seq.popleft()
                    self.freq_response_record.append(self._curr_step_record)
                    ret = Action(value=self._curr_action_seq.popleft())

        else:
            ret = Action(value=self._curr_action_seq.popleft())

        self._curr_step_record.action_seq.append(0. if ret.value is None else ret.value)
        self._curr_step_record.time_seq.append(obs.time - self._first_obs_time)
        self._curr_step_record.pos_seq.append(obs.pos)
        return ret

    # def step(self, obs:JointState)->Action:
    #     """
    #     None means pause this frame.
    #     """
    #
    #     if self._curr_action_seq is None:
    #         self._curr_freq, self._curr_action_seq = self.freq_action_seq.popleft()
    #         action = self._curr_action_seq.popleft()
    #         self._curr_step_record.freq = self._curr_freq
    #         # not save obs for head action.
    #         self._curr_step_record.action_seq.append(action)
    #         return Action(value=action)
    #
    #     elif len(self._curr_action_seq) == 0:
    #         if self._sleep_cnt == 0:
    #             # last obs of last action. no action to save.
    #             self._curr_step_record.time_seq.append(obs.time)
    #             self._curr_step_record.pos_seq.append(obs.pos)
    #
    #         if len(self.freq_action_seq) == 0:
    #             logger.warning(f'=== finish pop all the freq action seq. ===')
    #             return Action(value=None,last=True)
    #         else:
    #             # before start new freq action seq, sleep 3 sec.
    #             if self._sleep_cnt < self.sample_rate * 3:
    #                 self._sleep_cnt+=1
    #
    #                 # last obs of last action. no action to save.
    #                 self._curr_step_record.time_seq.append(obs.time)
    #                 self._curr_step_record.pos_seq.append(obs.pos)
    #                 self._curr_step_record.action_seq.append(0.)
    #
    #                 return Action(value=None)
    #
    #             else:
    #                 self._sleep_cnt = 0
    #                 # TODO: return None to sleep for 2sec?
    #                 self._curr_freq, self._curr_action_seq = self.freq_action_seq.popleft()
    #                 self._curr_step_record=StepRecord(freq=self._curr_freq,
    #                                                   action_seq=[],
    #                                                   time_seq=[],
    #                                                   pos_seq=[])
    #                 self._curr_step_record.freq = self._curr_freq
    #                 self.freq_response_record.append(self._curr_step_record)
    #                 action = self._curr_action_seq.popleft()
    #                 self._curr_step_record.action_seq.append(action)
    #                 return Action(value=action)
    #
    #     else:
    #         action = self._curr_action_seq.popleft()
    #         self._curr_step_record.action_seq.append(action)
    #         self._curr_step_record.time_seq.append(obs.time)
    #         self._curr_step_record.pos_seq.append(obs.pos)
    #         return Action(value=action)

    def term(self)->bool:
        return (len(self.freq_action_seq) == 0
                and len(self._curr_action_seq) == 0)

    def pre_process(self)->None:
        # set motor to zero point.
        pass


    def _plot_freq_response(self, plot_dir:Path):
        """
                绘制频率响应曲线

                Args:
                    plot_dir: 图像保存路径，若为None则显示图像
        """

        analyze_result:List[Dict[str, npt.NDArray[np.float32]]] = []

        for record in self.freq_response_record:
            # 分析频率响应
            input_fft = np.fft.rfft(record.action_seq)
            output_fft = np.fft.rfft(record.pos_seq)

            # 找到测试频率对应的FFT索引
            freq_idx = int(record.freq * self.dwell_time)

            # 计算幅值比和相位差
            input_mag = np.abs(input_fft[freq_idx])
            output_mag = np.abs(output_fft[freq_idx])
            phase_diff = np.angle(output_fft[freq_idx] / input_fft[freq_idx], deg=True)

            # 计算增益 (dB)
            gain_db = 20 * np.log10(output_mag / input_mag) if input_mag > 0 else -np.inf

            analyze_result.append({
                'frequency': record.freq,
                'gain_db': gain_db,
                'phase_deg': phase_diff,
                'input_mag': input_mag,
                'output_mag': output_mag
            })


        if not analyze_result:
            logger.error("没有频率响应数据，请先运行频率扫描测试")
            return

        frequencies = [data['frequency'] for data in self.analyze_result]
        gains_db = [data['gain_db'] for data in self.analyze_result]
        phases_deg = [data['phase_deg'] for data in self.analyze_result]

        # 计算带宽
        bandwidth = self.calculate_bandwidth(analyze_result)
        logger.warning(f"============ 伺服电机闭环带宽: {bandwidth:.2f} Hz ==================")

        plt.figure(figsize=(12, 8))

        # 绘制幅频特性曲线
        plt.subplot(2, 1, 1)
        plt.plot(frequencies, gains_db, 'b-', linewidth=2)
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

        if plot_dir:
            save_file = plot_dir / "actuation_{}_kp_{}_kd_{}.svg".format(time.strftime('%Y%m%d_%H%M%S'),
                                                                         DEFAULT_KP, DEFAULT_KD)
            plt.savefig(save_file, bbox_inches='tight')
            # plt.savefig(save_path.resolve().__str__(), dpi=300, bbox_inches='tight')
            # plt.savefig(save_file, dpi=300, bbox_inches='tight')
            logger.info(f"频率响应曲线已保存至: {plot_dir.resolve()}")
        else:
            plt.show()


    def _plot_joint_traj(self, plot_dir:Path):

        time_seq:List[float] = []
        pos_seq:List[float] = []
        action_seq:List[float] = []

        for _r in self.freq_response_record:
            time_seq.extend(_r.time_seq)
            action_seq.extend(_r.action_seq)
            pos_seq.extend(_r.pos_seq)

        print(f' =====  {len(time_seq)=:}  {len(action_seq)=:}  {len(pos_seq)=:} ===')


        plot_joint_tracking(
            {'0': time_seq },
            {'0': time_seq},
            {'0': pos_seq},
            {'0': action_seq},
            None,
            x_label="Time (s)",
            y_label="Pos (rad)",
            file_name="motor_pos_tracking",
            line_suffix=["_obs_pos", "_target_pos"],
            set_ylim=False,
            save_path=plot_dir.resolve().__str__(),
        )


    def post_process(self)->None:
        plot_dir = Path('./RS_motor_bandwidth_test_plot')
        if not plot_dir.exists():
            plot_dir.mkdir()

        self._plot_joint_traj(plot_dir)
        self._plot_freq_response(plot_dir)


    @staticmethod
    def calculate_bandwidth(analyze_result:List[Dict[str, npt.NDArray[np.float32]]],
                            low_freq_gain: float = None) -> float:
        """
        计算闭环带宽
        
        参数:
            low_freq_gain: 低频增益参考值，若为None则使用第一个频率点的增益
        
        返回:
            闭环带宽(Hz)
        """
        if not analyze_result:
            logger.error("没有频率响应数据，请先运行频率扫描测试")
            return 0.0
        
        # 确定参考增益 (通常是低频增益)
        if low_freq_gain is None:
            # 使用前5个频率点的平均增益作为参考
            # low_freq_points = min(5, len(self.freq_response_data))
            # use first point.
            low_freq_points = 1
            low_freq_gain = np.mean([data['gain_db'] for data in analyze_result[:low_freq_points]])
        
        # 计算-3dB点
        target_gain = low_freq_gain - 3.0

        logger.info(f'{target_gain=:}')
        
        # 找到第一个低于目标增益的频率点
        bandwidth = 0.0
        for i, data in enumerate(analyze_result):
            if data['gain_db'] < target_gain:
                if i > 0:
                    # 线性插值以获得更准确的带宽估计
                    prev_data = analyze_result[i-1]
                    x0, y0 = prev_data['frequency'], prev_data['gain_db']
                    x1, y1 = data['frequency'], data['gain_db']
                    
                    # 插值公式: x = x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)
                    bandwidth = x0 + (target_gain - y0) * (x1 - x0) / (y1 - y0)
                else:
                    bandwidth = data['frequency']
                break
        
        # 如果没有找到低于-3dB的点，返回最高测试频率
        if bandwidth == 0.0 and analyze_result:
            bandwidth = analyze_result[-1]['frequency']
        
        logger.info(f"计算得到闭环带宽: {bandwidth:.2f}Hz")
        return bandwidth



if __name__ == "__main__":
    pass