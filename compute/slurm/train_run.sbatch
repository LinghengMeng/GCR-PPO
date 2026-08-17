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
# Optional:
#   RESUME_SRC_CHECKPOINT   absolute path to a model_N.pt from an earlier run to continue training
#     from (2026-08-18, checking whether GCR-PPO's own curves - still rising at the end of the
#     first ~22-24k-iteration sweep, unlike the parallel HraPPO sweep which plateaued early - keep
#     improving with more training). Must follow the standard
#     .../logs/rsl_rl/<experiment_name>/<run_dir>/model_N.pt layout (GCR-PPO's train.py also saves
#     a copy under wandb/offline-run-*/files/, but get_checkpoint_path() only looks in the standard
#     logs/rsl_rl location, so that's the one to point at). Each job runs in a fresh per-job scratch
#     directory, so the source checkpoint can't be resolved relative to THIS job's own
#     log_root_path - copied into place under this job's SCRATCH_RUN before training starts.
set -uo pipefail

: "${CONDITION:?CONDITION must be set (naive_multihead|conflict_resolved)}"
: "${SEED:?SEED must be set}"

CODE_DIR="/datastore/${USER}/hra_gcr_code"
SANDBOX="/scratch3/${USER}/gcr_ppo/isaac_sim_gcr_sandbox"
REPO="${CODE_DIR}/GCR-PPO"
PYTHONUSERBASE_GCR="${CODE_DIR}/pyuserbase_gcr"

RUN_DIR="gcr_ppo_${CONDITION}_seed${SEED}_${SLURM_JOB_ID}"
SCRATCH_RUN="/scratch3/${USER}/gcr_ppo_runs/${RUN_DIR}"
DATASTORE_RUN="/datastore/${USER}/gcr_ppo_runs/${RUN_DIR}"

mkdir -p "${SCRATCH_RUN}"

sync_to_datastore() {
    mkdir -p "$(dirname "${DATASTORE_RUN}")"
    # rsync, not cp -r: cp -r into an already-existing DATASTORE_RUN (i.e. every sync after the
    # first) nests the copy one level deeper instead of overwriting in place - confirmed via the
    # parallel men119-isaaclabextensioncsirohri sweep 2026-08-13. rsync -a is idempotent.
    rsync -a "${SCRATCH_RUN}/" "${DATASTORE_RUN}/" 2>/dev/null || true
}

# Periodic sync every 5 min (tightened from 20 min 2026-08-13) - the --signal=B:SIGTERM@180 EXIT
# trap is NOT reliable: confirmed on the parallel men119-isaaclabextensioncsirohri sweep that the
# trap can simply not fire before the hard time-limit kill, even though SLURM still reports a
# clean exit. GCR-PPO's checkpoints live ONLY under SCRATCH_RUN (wandb/offline-run-*/files/,
# unlike the other repo which also writes checkpoints directly to /datastore independently), so
# for these jobs the periodic sync is the ONLY real safety net - do not rely on the trap.
( while true; do sleep 300; sync_to_datastore; done ) &
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

RESUME_ARGS=()
if [ -n "${RESUME_SRC_CHECKPOINT:-}" ]; then
    RESUME_RUN_DIR="$(basename "$(dirname "${RESUME_SRC_CHECKPOINT}")")"
    RESUME_EXPERIMENT_NAME="$(basename "$(dirname "$(dirname "${RESUME_SRC_CHECKPOINT}")")")"
    RESUME_CHECKPOINT_FILE="$(basename "${RESUME_SRC_CHECKPOINT}")"
    RESUME_DEST_DIR="${SCRATCH_RUN}/logs/rsl_rl/${RESUME_EXPERIMENT_NAME}/${RESUME_RUN_DIR}"
    mkdir -p "${RESUME_DEST_DIR}"
    cp "${RESUME_SRC_CHECKPOINT}" "${RESUME_DEST_DIR}/"
    RESUME_ARGS=(--resume --load_run "${RESUME_RUN_DIR}" --checkpoint "${RESUME_CHECKPOINT_FILE}")
    echo "Resuming from: ${RESUME_SRC_CHECKPOINT} (copied to ${RESUME_DEST_DIR}/${RESUME_CHECKPOINT_FILE})"
fi

echo "Run: ${RUN_DIR}  (condition=${CONDITION} seed=${SEED} job=${SLURM_JOB_ID})"
echo "DEBUG ALG_FLAGS: [${ALG_FLAGS[@]}]  (count: ${#ALG_FLAGS[@]})"
echo "DEBUG RESUME_ARGS: [${RESUME_ARGS[@]}]  (count: ${#RESUME_ARGS[@]})"

module load apptainer

# Write the inner command to a real script file instead of embedding array expansions in a
# double-quoted `bash -c "..."` argument passed through apptainer exec. Found 2026-08-18: with
# CONDITION=conflict_resolved (ALG_FLAGS has 2 elements, plus RESUME_ARGS with 4 more), sys.argv
# inside the container showed the argument list silently truncated right after the first ALG_FLAGS
# element - --use_pcgrad and the entire RESUME_ARGS were dropped before reaching Python, even
# though an echo of the same arrays immediately before this block showed them fully correct.
# naive_multihead (ALG_FLAGS has only 1 element) never hit this, which is why it looked like an
# isolated pcgrad-specific bug at first. Root mechanism not fully pinned down (suspect something
# in how apptainer/the container boundary handles a long multi-element single-argument string),
# but writing to a file and executing that sidesteps it entirely - confirmed the more robust
# approach after `sys.argv` debugging on the previous (broken) inline version.
INNER_SCRIPT="${SCRATCH_RUN}/_run_inner.sh"
cat > "${INNER_SCRIPT}" <<INNEREOF
set -uo pipefail
export TERM=xterm
export PATH=\$PATH:/usr/bin
export PYTHONUSERBASE='${PYTHONUSERBASE_GCR}'
export WANDB_MODE=offline
export WANDB_USERNAME='${USER}'
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd '${SCRATCH_RUN}'
'${REPO}/_isaac_sim/python.sh' '${REPO}/scripts/reinforcement_learning/rsl_rl/train.py' \\
    --task Isaac-Lift-Mug-Aloha-Right-Arm-v0 \\
    --num_envs 200 \\
    --seed ${SEED} \\
    --headless \\
    ${ALG_FLAGS[@]} \\
    ${RESUME_ARGS[@]} \\
    2>&1 | tee '${SCRATCH_RUN}/train.log'
INNEREOF
echo "DEBUG inner script contents:"
cat "${INNER_SCRIPT}"

apptainer exec --nv --writable "${SANDBOX}" bash "${INNER_SCRIPT}"

echo "Training command exited (code ${PIPESTATUS[0]}) - EXIT trap will now sync scratch -> datastore."
