# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from Lingheng Meng's men119-isaaclabextensioncsirohri
# (exts/pref_hidden_values/pref_hidden_values/assets/aloha_right_arm.py), namespace-only
# translation (omni.isaac.lab.* -> isaaclab.*) for the GCR-PPO comparison experiment - see
# vault note Ongoing_2026-02-05_Hybrid_Reward_PPO/2026-08-12_Multihead_Conflict_Resolution_Investigation.md.
# No other changes - same joint init state, actuator gains, USD asset.

"""Configuration for the ALOHA right-arm robot (single-arm ALOHA)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# USD/texture assets copied alongside this task (see ../../aloha_mug_assets/, i.e. lift/aloha_mug_assets/),
# not the shared isaaclab_assets nucleus - this robot/object pair is specific to this experiment.
# NOTE: this file lives at lift/config/aloha_right_arm/ - three levels below lift/, so three
# dirname() calls are needed to reach it. Had only two here originally (landed at lift/config/
# instead of lift/), causing FileNotFoundError for the robot USD - confirmed via smoke test
# 2026-08-13 (joint_pos_env_cfg.py's mug asset path, computed differently via "..", ".." segments,
# was already correct).
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "aloha_mug_assets")

##
# Configuration
##

ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_DIR, "aloha", "aloha_right_arm.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=[0.46, 0, 0], rot=[0, 0, 0, 1],
        joint_pos={
            # right arm - joint positions in radians (Isaac Sim GUI shows degrees)
            "right_waist": 0.0,
            "right_shoulder": 0.5393,  # 30.90 deg
            "right_elbow": 0.1,  # 5.73 deg
            "right_forearm_roll": 0.0,
            "right_wrist_angle": 1.0297,  # 59.00 deg
            "right_wrist_rotate": 0.0,
            "right_left_finger": 0.041,
            "right_right_finger": 0.041,
        },
    ),
    actuators={
        "right_waist": ImplicitActuatorCfg(
            joint_names_expr=["right_waist"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_elbow": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_forearm_roll": ImplicitActuatorCfg(
            joint_names_expr=["right_forearm_roll"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_wrist_angle": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist_angle"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_wrist_rotate": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist_rotate"], effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0
        ),
        "right_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_left_finger", "right_right_finger"],
            effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
