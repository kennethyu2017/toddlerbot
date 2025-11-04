"""
follow the mujoco_playground.
"""
from ml_collections import config_dict

def _task_to_xml(task_name: str) -> str:
  return {
      "flat_terrain": 'scene_feetonly_flat_terrain_mjx.xml',
      "rough_terrain": 'scene_feetonly_rought_terrain_mjx.xml',
  }[task_name]

#nconmax,njmax are deprecated only use for mujoco prior to 2.3.0.,
# and only used for `warp` backend.
# def _task_to_nconmax(task_name: str) -> int:
#     return {
#         'flat_terrain': 8 * 8192,
#         'rough_terrain': 100 * 8192,
#     }[task_name]
#
# def _task_to_njmax(task_name: str) -> int:
#     return {
#         'flat_terrain': 29 * 2 + 8 * 4,
#         'rough_terrain':  29 * 2 + 100 * 4,
#     }[task_name]

def _model_config(task_name:str) -> config_dict.ConfigDict:
    return config_dict.create(
        # relative to project dir.
        xml_dir='kbot/locomotion/kbot_both_leg/xmls',
        xml_file=_task_to_xml(task_name),
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        impl="jax",
        #nconmax,njmax are deprecated only use for mujoco prior to 2.3.0.
        # and only used for `warp` backend.
        # nconmax=_task_to_nconmax(task_name),
        # njmax=_task_to_njmax(task_name),
        noise=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,
            ),
        ),
    )

def _robot_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # joint
        joints=config_dict.create(
            free_joint='floating_base_joint',
            hip_pitch_joints=['left_hip_pitch_RS04', 'right_hip_pitch_RS04'],
            hip_roll_joints=['left_hip_roll_RS03', 'right_hip_roll_RS03'],
            hip_yaw_joints=['left_hip_yaw_RS03','right_hip_yaw_RS03'],
            knee_pitch_joints=['left_knee_pitch_RS04', 'right_knee_pitch_RS04'],
            ankle_pitch_joints=['left_ankle_pitch_passive_on_foot_structure_RS02',
                                'right_ankle_pitch_passive_on_foot_structure_RS02'],
        ),
        # site
        sites=config_dict.create(
            feet_sites=["left_foot_ankle", "right_foot_ankle"],
            pelvis_imu_site='imu_in_pelvis',
        ),
        # geom
        geoms=config_dict.create(
            feet_collision_geoms=['left_foot_collision', 'right_foot_collision'],
        ),
        # sensor
        #sensor names will be combined with _frame, e.g., upvector_torso.
        sensors=config_dict.create(
            upvector_pelvis='upvector_pelvis',
            forwardvector_pelvis='forwardvector_pelvis',
            orientation_pelvis='orientation_pelvis',
            global_linvel_pelvis='global_linvel_pelvis',
            global_angvel_pelvis='global_angvel_pelvis',

            # IMU data
            gyro_pelvis='gyro_pelvis',
            accelerometer_pelvis='accelerometer_pelvis',
            local_linvel_pelvis='local_linvel_pelvis',

            # foot vel
            global_linvel_feet_ankle=['global_linvel_left_foot_ankle','global_linvel_right_foot_ankle'],
            upvector_left_feet_ankle=['upvector_left_foot_ankle','upvector_right_foot_ankle'],

            # force
            feet_force=['left_foot_force', 'right_foot_force'],

            # contact
            floor_feet_found=['floor_left_foot_found', 'floor_right_foot_found'],
            left_leg_right_leg_found= ['left_foot_right_foot_found', 'left_foot_right_shin_found', 'left_foot_right_thigh_found',
                                       'left_shin_right_foot_found', 'left_shin_right_shin_found', 'left_shin_right_thigh_found',
                                       'left_thigh_right_foot_found', 'left_thigh_right_shin_found', 'left_thigh_right_thigh_found'],
        ),
        # keyframe
        keyframes=config_dict.create(
            default_pose_keyframe='knee_bent',
        ),
        # restricted_joint_range = (
        #     # Left leg.
        #     (-1.57, 1.57),
        #     (-0.5, 0.5),
        #     (-0.7, 0.7),
        #     (0, 1.57),
        #     (-0.4, 0.4),
        #     (-0.2, 0.2),
        #     # Right leg.
        #     (-1.57, 1.57),
        #     (-0.5, 0.5),
        #     (-0.7, 0.7),
        #     (0, 1.57),
        #     (-0.4, 0.4),
        #     (-0.2, 0.2),
        # )
    )

def _rwd_config() -> config_dict.ConfigDict:
    return config_dict.create(
        scales=config_dict.create(
            # Tracking related rewards.
            tracking_lin_vel=1.0,
            tracking_ang_vel=0.75,
            # Base related rewards.
            lin_vel_z=0.0,
            ang_vel_xy=-0.15,
            orientation=-2.0,
            base_height=0.0,
            # Energy related rewards.
            torques=0.0,
            action_rate=0.0,
            energy=0.0,
            dof_acc=0.0,
            # Feet related rewards.
            feet_clearance=0.0,
            feet_air_time=2.0,
            feet_slip=-0.25,
            feet_height=0.0,
            feet_phase=1.0,
            # Other rewards.
            alive=0.0,
            stand_still=-1.0,
            termination=-100.0,
            collision=-0.1,
            contact_force=-0.01,
            # Pose related rewards.
            joint_deviation_knee=-0.1,
            joint_deviation_hip=-0.25,
            dof_pos_limits=-1.0,
            pose=-0.1,
        ),
        tracking_sigma=0.25,
        max_foot_height=0.15,
        base_height_target=0.5,
        max_contact_force=500.0,
    )

def _push_config() -> config_dict.ConfigDict:
    return config_dict.create(
        enable=True,
        # in second.
        interval_range=[5.0, 10.0],

        magnitude_range=[0.1, 2.0],
    )

def _cmd_config() -> config_dict.ConfigDict:
    return config_dict.create(
          # cmd range
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.5, 0.5],
          ang_vel_yaw=[-1.0, 1.0],

          # Uniform distribution for command amplitude.
          a=[1.0, 0.8, 1.0],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      )

def default_config(task_name:str) -> config_dict.ConfigDict:
  return config_dict.create(
      model=_model_config(task_name),
      robot=_robot_config(),
      reward=_rwd_config(),
      push=_push_config(),
      command=_cmd_config(),
  )


if __name__ == '__main__':
    print(default_config('flat_terrain'))
