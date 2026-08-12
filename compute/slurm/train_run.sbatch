#!/bin/bash
#SBATCH --account=OD-241146
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGTERM@180
#SBATCH --output=/scratch3/%u/gcr_ppo_runs/slurm_%j.log
#
# Single training run of GCR-PPO's OWN implementation (not the ported HraPPO conflict-resolution
# code - this runs their actual repo/algorithm) on the ALOHA lift-mug task, for a direct
# comparison against the parallel multi-head-PPO conflict-resolution experiment (see vault note
# Ongoing_2026-02-05_Hybrid_Reward_PPO/2026-08-12_Multihead_Conflict_Resolution_Investigation.md).
#
# GCR-PPO is built on IsaacLab 2.1.0, which needs Isaac Sim installed via pip (isaacsim-rl) -
# doesn't work on Virga at all: the wheels require manylinux_2_34 (glibc >= 2.34), Virga runs
# glibc 2.31 on every node (login and compute, confirmed 2026-08-13). Runs via Apptainer instead,
# using the official nvcr.io/nvidia/isaac-sim:4.5.0 container (Ubuntu 22.04, glibc 2.35) - NOT the
# same Apptainer path that hung earlier this session with a different (old, decayed) container;
# this one confirmed working cleanly, including --nv, via direct testing.
#
# Uses the WRITABLE SANDBOX (not the read-only .sif) consistently, matching how it was set up:
# isaaclab.sh -i had to uninstall/replace the container's own bundled torch, which only happened
# in the sandbox copy - running training against the original read-only .sif would find the old,
# removed-in-the-sandbox torch still present and get an inconsistent environment.
#
# Required env vars (set via --export by submit_sweep.sh):
#   CONDITION   one of: naive_multihead | conflict_resolved
#     (no separate "baseline" here - already have the scalar-PPO baseline from the main sweep;
#     this comparison is specifically multi-head-with-vs-without-conflict-resolution, using
#     GCR-PPO's own code, not a full re-run of every condition)
#   SEED        integer seed
set -uo pipefail

: "${CONDITION:?CONDITION must be set (naive_multihead|conflict_resolved)}"
: "${SEED:?SEED must be set}"

CODE_DIR="/datastore/${USER}/hra_gcr_code"
SANDBOX="${CODE_DIR}/isaac_sim_gcr_sandbox"
REPO="${CODE_DIR}/GCR-PPO"
PYTHONUSERBASE_GCR="${CODE_DIR}/pyuserbase_gcr"

RUN_DIR="gcr_ppo_${CONDITION}_seed${SEED}_${SLURM_JOB_ID}"
SCRATCH_RUN="/scratch3/${USER}/gcr_ppo_runs/${RUN_DIR}"
DATASTORE_RUN="/datastore/${USER}/gcr_ppo_runs/${RUN_DIR}"

mkdir -p "${SCRATCH_RUN}"

sync_to_datastore() {
    mkdir -p "$(dirname "${DATASTORE_RUN}")"
    cp -r "${SCRATCH_RUN}" "${DATASTORE_RUN}" 2>/dev/null || true
}

( while true; do sleep 1200; sync_to_datastore; done ) &
PERIODIC_SYNC_PID=$!
trap 'kill "${PERIODIC_SYNC_PID}" 2>/dev/null; echo "Final sync to datastore..."; sync_to_datastore; echo "Done."' EXIT

case "${CONDITION}" in
  naive_multihead)
    ALG_FLAGS=(--use_critic_multi)
    ;;
  conflict_resolved)
    ALG_FLAGS=(--use_critic_multi --use_pcgrad)
    ;;
  *)
    echo "Unknown CONDITION: ${CONDITION}" >&2
    exit 1
    ;;
esac

echo "Run: ${RUN_DIR}  (condition=${CONDITION} seed=${SEED} job=${SLURM_JOB_ID})"

module load apptainer

apptainer exec --nv --writable "${SANDBOX}" bash -c "
set -uo pipefail
export TERM=xterm
export PYTHONUSERBASE='${PYTHONUSERBASE_GCR}'
export WANDB_MODE=offline
cd '${SCRATCH_RUN}'
'${REPO}/_isaac_sim/python.sh' '${REPO}/scripts/reinforcement_learning/rsl_rl/train.py' \
    --task Isaac-Lift-Mug-Aloha-Right-Arm-v0 \
    --num_envs 200 \
    --seed ${SEED} \
    --headless \
    ${ALG_FLAGS[@]} \
    2>&1 | tee '${SCRATCH_RUN}/train.log'
"

echo "Training command exited (code ${PIPESTATUS[0]}) - EXIT trap will now sync scratch -> datastore."
