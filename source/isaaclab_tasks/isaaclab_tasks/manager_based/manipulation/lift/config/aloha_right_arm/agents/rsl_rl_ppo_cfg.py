# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Hyperparameters matched to the multi-head-PPO conflict-resolution experiment's own HRA agent
# configs (men119-isaaclabextensioncsirohri, aloha_right_arm/agents/test_rsl_rl_alg_branch_ppo_hra_gcr_cfg/)
# wherever this config surface has a corresponding field, for a fair comparison. Not a perfect
# match - GCR-PPO's OnPolicyRunner controls the multi-head critic structure internally
# (num heads = env_cfg.reward_components) rather than via a nested critic_hidden_dims shape like
# the other codebase, so critic_hidden_dims here is the flat per-head width, closest analogue
# available. --use_critic_multi/--use_pcgrad (CLI flags, not part of this cfg) select the
# baseline/naive-multihead/conflict-resolved condition at launch time.
# max_iterations left high (matches the other experiment's convention) - every run is expected to
# be stopped by the SLURM job's --time limit, not finish naturally.

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftMugAlohaGcrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 1000000
    save_interval = 50
    experiment_name = "aloha-right-arm_lift_mug_gcr_ppo"
    logger = "wandb"
    wandb_project = experiment_name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
