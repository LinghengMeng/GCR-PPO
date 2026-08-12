# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from Lingheng Meng's men119-isaaclabextensioncsirohri
# (exts/pref_hidden_values/pref_hidden_values/tasks/manager_based/manipulation/lift/config/aloha_right_arm/joint_pos_env_cfg.py),
# namespace-only translation (omni.isaac.lab.* -> isaaclab.*) for the GCR-PPO comparison
# experiment - same reward decomposition (6 terms) as the multi-head-PPO conflict-resolution
# experiment this compares against. See vault note
# Ongoing_2026-02-05_Hybrid_Reward_PPO/2026-08-12_Multihead_Conflict_Resolution_Investigation.md.
#
# reward_components/reward_component_names/reward_component_task_rew follow the exact pattern
# GCR-PPO's own ../franka/joint_pos_env_cfg.py already uses for this same reward decomposition -
# not a new design choice, matching their existing convention.

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from .aloha_right_arm_cfg import ALOHA_CFG  # isort: skip

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "aloha_mug_assets")


@configclass
class AlohaMugLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Aloha as robot
        self.scene.robot = ALOHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (aloha) - scale=1, use_default_offset=False:
        # processed_action = raw_action * scale + offset, raw_action in [-1,1]. Matches the
        # original config exactly (the alternative per-joint scale/offset in the original's
        # commented-out block was noted there as "works bad" - not carried over).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
                         "right_wrist_angle", "right_wrist_rotate"],
            scale=1,
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_left_finger", "right_right_finger"],
            open_command_expr={"right_left_finger": 0.041, "right_right_finger": 0.041},
            close_command_expr={"right_left_finger": 0.0, "right_right_finger": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "right_gripper_base"

        # Set Mug as object (Aloha gripper is smaller than Franka Panda, so the mug is scaled
        # down accordingly - matches the original config)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(_ASSETS_DIR, "mug", "mug_collider_triangle_mesh.usd"),
                scale=(0.005, 0.005, 0.005),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        # Multi-objective reward decomposition, for GCR-PPO's --use_critic_multi/--use_pcgrad -
        # same 6 terms, same task/penalty split, as the multi-head-PPO conflict-resolution
        # experiment this compares against. Pattern matches ../franka/joint_pos_env_cfg.py.
        self.reward_components = 6
        self.reward_component_names = [
            "reaching_object", "lifting_object", "object_goal_tracking",
            "object_goal_tracking_fine_grained", "action_rate", "joint_vel",
        ]
        self.reward_component_task_rew = [
            "reaching_object", "lifting_object", "object_goal_tracking", "object_goal_tracking_fine_grained",
        ]


@configclass
class AlohaMugLiftEnvCfg_PLAY(AlohaMugLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
