# Virga job scripts — running GCR-PPO's own implementation on the ALOHA lift-mug task

Runs GCR-PPO's *actual, unmodified* algorithm (not a port) on the ALOHA lift-mug task added in
this branch (`source/isaaclab_tasks/.../lift/config/aloha_right_arm/`), for direct comparison
against the parallel multi-head-PPO conflict-resolution experiment's own results. See vault note
`Ongoing_2026-02-05_Hybrid_Reward_PPO/2026-08-12_Multihead_Conflict_Resolution_Investigation.md`.

## Why Apptainer, and why a specific container

GCR-PPO needs IsaacLab 2.1.0, which needs Isaac Sim installed via pip (`isaacsim-rl`). That
doesn't work on Virga at all: the wheels require `manylinux_2_34` (glibc >= 2.34), and Virga runs
glibc 2.31 on every node - login and compute, confirmed 2026-08-13. Runs via Apptainer instead,
using the official `nvcr.io/nvidia/isaac-sim:4.5.0` image (Ubuntu 22.04, glibc 2.35).

This is a **different** Apptainer path than the one that hung earlier in this investigation (see
the main sweep's job scripts) - that was a different, decayed container from disuse. This one was
tested directly (including `--nv`) before committing to the full setup and works cleanly.

## Setup (already done once for this session, documented for reproducibility)

```bash
module load apptainer
cd /datastore/$USER/hra_gcr_code
apptainer pull isaac_sim_4.5.0.sif docker://nvcr.io/nvidia/isaac-sim:4.5.0

# Container filesystem is read-only by default - isaaclab.sh -i needs to replace the
# container's own bundled torch, so build a writable sandbox:
apptainer build --sandbox --nv isaac_sim_gcr_sandbox isaac_sim_4.5.0.sif

# Virga's apptainer module auto-binds /apps, /datasets, /scratch3, /datastore into every
# container (via $APPTAINER_BINDPATH) - in --writable mode Apptainer can't auto-create missing
# mountpoints for these (only non-writable/overlay mode can), so create them once:
mkdir -p isaac_sim_gcr_sandbox/{apps,datasets,scratch3,datastore}

cd GCR-PPO
ln -sfn /isaac-sim _isaac_sim   # only resolves correctly from inside the container

apptainer exec --writable /datastore/$USER/hra_gcr_code/isaac_sim_gcr_sandbox bash -c '
    export TERM=xterm   # isaaclab.sh errors on an unset/unknown terminal type otherwise
    export PYTHONUSERBASE=/datastore/'"$USER"'/hra_gcr_code/pyuserbase_gcr
    cd /datastore/'"$USER"'/hra_gcr_code/GCR-PPO
    ./isaaclab.sh -i
    # isaaclab_mimic fails here (needs git, not in this container) - fine, not needed for this task
    ./isaaclab.sh -p -m pip uninstall -y rsl-rl-lib
    ./isaaclab.sh -p -m pip install -e rsl_rl
'
```

**Uses the writable sandbox consistently, not the read-only `.sif`**, for both setup and actual
training runs: `isaaclab.sh -i` uninstalled/replaced the container's own bundled torch, which only
happened in the sandbox copy - running against the original read-only `.sif` would find the old,
still-present-there torch and get an inconsistent environment.

## Usage

```bash
bash compute/slurm/submit_sweep.sh
```

Two conditions x 3 seeds (no separate scalar-PPO baseline - already have that from the main
sweep's `hra_gcr_code/men119-isaaclabextensioncsirohri` runs):

| Condition | Flags |
|---|---|
| `naive_multihead` | `--use_critic_multi` (multi-head critic, no conflict resolution) |
| `conflict_resolved` | `--use_critic_multi --use_pcgrad` (GCR-PPO's full method) |

Same scratch-then-datastore pattern as the main sweep: writes to `/scratch3/$USER/gcr_ppo_runs/<run_dir>`
during training (fast), synced to `/datastore/$USER/gcr_ppo_runs/<run_dir>` periodically (every 20
min) and on exit (normal completion, error, or the SIGTERM SLURM sends 180s before the `--time`
limit kill - every run is expected to hit that, `max_iterations` isn't bounded). `WANDB_MODE=offline`
(Virga compute nodes have no internet) - sync from the login node afterward:
```bash
wandb sync /datastore/$USER/gcr_ppo_runs/<run_dir>/wandb/offline-run-*
```
