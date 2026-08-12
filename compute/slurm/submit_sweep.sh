#!/bin/bash
# Submits GCR-PPO's own naive_multihead (--use_critic_multi) and conflict_resolved
# (--use_critic_multi --use_pcgrad) conditions on the ALOHA lift-mug task, for direct comparison
# against the parallel multi-head-PPO conflict-resolution experiment's own HraPPO-based results.
# No separate scalar-PPO baseline here - already have that from the main sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDITIONS=(naive_multihead conflict_resolved)
SEEDS=(1 2 3)

for CONDITION in "${CONDITIONS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    JOB_NAME="gcr_ppo_${CONDITION}_seed${SEED}"
    echo "Submitting ${JOB_NAME}..."
    sbatch --job-name="${JOB_NAME}" \
           --export=CONDITION="${CONDITION}",SEED="${SEED}" \
           "${SCRIPT_DIR}/train_run.sbatch"
  done
done
