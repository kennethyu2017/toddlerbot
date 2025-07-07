import struct
import numpy as np
from numpy import typing as npt
from typing import Tuple, Dict,Type

from .._module_logger import logger

class ParamConverter:

    _StructFmtStr: Dict[Tuple[int, Type, bool], str] = {
        # key is: (n_bytes, dtype, signed)
        # always little endian for RobStride.
        (4, float, True): '<f',   # float32: IEEE-754 binary32 single precision float.
        (4, int, False): '<I',    # uint32
        (2, int, False): '<H',    # uint16
        (1, int, False): '<B',    # uint8
    }

    @staticmethod
    def float_normalized_to_uint(*, x:float|npt.NDArray[np.float32],
                      x_min:float|npt.NDArray[np.float32],
                      x_max:float|npt.NDArray[np.float32],
                      n_bytes:int)->np.ndarray[np.uint32]:
        """
        将浮点数 normalized 为无符号整数。

        参数:
        x: 输入的浮点数。
        x_min: 可接受的最小浮点数。
        x_max: 可接受的最大浮点数。
        n_bytes: 输出无符号整数的位数。

        返回:
        转换后的无符号整数。
        """
        assert n_bytes <= 4

        float_value_span = x_max - x_min
        target_uint_span = (1 << (n_bytes*8) ) - 1

        # x = max(min(x, x_max), x_min)  # Clamp x to the range [x_min, x_max]

        clamped_x = np.where(
            x > x_max,  # Condition 1
            x_max,  # Value when condition 1 is True
            np.where(
                x < x_min,  # Condition 2
                x_min,
                x
            ),
        )

        return ((clamped_x - x_min) * target_uint_span / float_value_span).astype(np.uint32)

    @staticmethod
    def uint_normalized_to_float(*, x:int|npt.NDArray[np.uint32],
                      x_min:float|npt.NDArray[np.float32],
                      x_max:float|npt.NDArray[np.float32],
                      n_bytes:int)->np.ndarray[np.float32]|float:
        """
        将无符号整数转换为浮点数。

        参数:
        x: 输入的无符号整数。
        x_min: 可接受的最小浮点数。
        x_max: 可接受的最大浮点数。
        n_bytes: 输入无符号整数的位数。

        返回:
        转换后的浮点数。
        """
        assert n_bytes <= 4

        uint_value_span = (1 << (n_bytes*8) ) - 1
        target_float_span = x_max - x_min

        # x = max(min(x, span), 0)  # Clamp x to the range [0, span]

        clamped_x = np.where(
            x > uint_value_span,  # Condition 1
            uint_value_span,  # Value when condition 1 is True
            np.where(
                x < 0,  # Condition 2
                0,
                x
            ),
        )

        return (target_float_span * clamped_x / uint_value_span + x_min).astype(np.float32)

    @staticmethod
    def linear_mapping(*,
            value:npt.NDArray[np.float32],
            value_min:float|int,
            value_max:float|int,
            target_min=0,
            target_max=65535)->np.ndarray[np.uint32]:
        """
        对输入值进行线性映射。

        参数:
        value: 输入值。
        value_min: 输入值的最小界限。
        value_max: 输入值的最大界限。
        target_min: 输出值的最小界限。
        target_max: 输出值的最大界限。

        返回:
        映射后的值。
        """
        assert 0 < (target_max-target_min) < 65535

        value_span = value_max - value_min
        target_span = target_max - target_min

        return (
                (value - value_min) * target_span / value_span
            + target_min
        ).astype(np.uint32)


    @staticmethod
    def _param_bytes_to_float(param: bytes)->float:
        # convert IEEE-754 binary32 single-precision hex bytes, received from can bus, to float value.
        assert len(param)==4
        try:
            # little endian, binary32 4-bytes single-precision float.
            value_tuple : Tuple[float] = struct.unpack('<f', param)
            assert len(value_tuple) == 1

        except Exception as exc:
            logger.error(f'param bytes unpack error: {param=:} {exc=:} {type(exc)=:}')
            raise exc

        return value_tuple[0]

    @staticmethod
    def _float_to_param_bytes(value: float)->bytes:
        # convert float to IEEE-754 binary32 single-precision hex bytes, to be sent to can bus.
        try:
            # little endian, binary32 4-bytes single-precision float.
            param:bytes = struct.pack('<f', value)
            assert len(param) == 4

        except Exception as exc:
            logger.error(f'pack float to param bytes error: {value=:} {exc=:} {type(exc)=:}')
            raise exc

        return param

    @staticmethod
    def value_to_param_bytes(*,value:float|int,
                             n_bytes:int,
                             dtype:Type,
                             signed:bool)->bytes:
        # key is: (n_bytes, dtype, signed)
        try:
            fmt_str = ParamConverter._StructFmtStr[(n_bytes,dtype,signed)]
        except KeyError as exc:
            logger.error(f'can not find supported struct format str: {exc=:} {type(exc)=:} ')
            raise exc

        try:
            param: bytes = struct.pack(fmt_str, value)
            assert len(param) == n_bytes

        except Exception as exc:
            logger.error(f'pack value to param bytes error: {value=:} {exc=:} {type(exc)=:}')
            raise exc

        return param

    @staticmethod
    def param_bytes_to_value(*, param: bytes | bytearray,
                             n_bytes: int,
                             dtype: Type,
                             signed: bool) -> float|int:
        # key is: (n_bytes, dtype, signed)
        try:
            fmt_str = ParamConverter._StructFmtStr[(n_bytes, dtype, signed)]
        except KeyError as exc:
            logger.error(f'can not find supported struct format str: {exc=:} {type(exc)=:} ')
            raise exc

        try:
            value:int|float = struct.unpack(fmt_str, param)[0]

        except Exception as exc:
            logger.error(f'unpack param bytes to value error: {param=:} {exc=:} {type(exc)=:}')
            raise exc

        return value

