from typing import Tuple
from can import Message

from .._module_logger import logger
from .robstride_def import _ParamTableReadSpec,CommunicationType

# for RobStride private protocol process.
# void RobStrite_Get_CAN_ID();
# 	void Set_RobStrite_Motor_parameter(uint16_t Index, float Value, char Value_mode);
# 	void Get_RobStrite_Motor_parameter(uint16_t Index);
# 	void RobStrite_Motor_Analysis(uint8_t *DataFrame,uint32_t ID_ExtId);
# 	void RobStrite_Motor_move_control(float Torque, float Angle, float Speed, float Kp, float Kd);
# 	void RobStrite_Motor_Pos_control( float Speed,float acceleration, float Angle);
# 	void RobStrite_Motor_Speed_control(float Speed,float acceleration, float limit_cur);
# 	void RobStrite_Motor_current_control( float current);
# 	void RobStrite_Motor_Set_Zero_control();
# 	void Enable_Motor();
# 	void Disenable_Motor( uint8_t clear_error);
# 	void Set_CAN_ID(uint8_t Set_CAN_ID);
# 	void Set_ZeroPos();

class RSProtocolBuilder:

    @staticmethod
    def get_device_id(motor_can_id: int) -> Message:
        assert 0 <= motor_can_id <= 0xff  # 1-byte

        # Calculate the arbitration ID
        arbitration_id = (CommunicationType.GET_DEVICE_ID << 24) | (data2 << 8) | motor_can_id
        try:
            # will check arbitration ID in can.Message.
            msg = Message(
                arbitration_id=arbitration_id,
                dlc=8,
                data=data1,
                is_extended_id=True
            )
        except Exception as exc:
            logger.error(f'build can msg error: {exc=:} {type(exc)=:}')
            raise exc
        return msg

    @staticmethod
    def motion_control():
        pass
    



class RSProtocolParser:

    @staticmethod
    def motor_feedback():
        pass
