#!/bin/bash
# Shared body for the LeWM baseline jobs. Not submitted directly: each slurm/train_<env>.sh
# sets the #SBATCH resources plus RUN_NAME / DATA_CFG / DATA_SRC and then execs this file.
#
# Layout (everything on scratch; home is at quota):
#   /scratch/ad158/baselines/le-wm/
#     checkpoints/<RUN_NAME>/   weights_epoch_N.pt, config.json, config.yaml
#     spt/                      Lightning run dirs with last.ckpt (used for requeue resume)
#     hydra/<RUN_NAME>/         hydra outputs
#     runs/<RUN_NAME>/          per-run working dir: local wandb files + wandb_resume.json
#     logs/                     slurm stdout/stderr
# W&B: ahitagnied-rice-university/baselines, run name = RUN_NAME, group = le-wm.

set -euo pipefail
: "${RUN_NAME:?}" "${DATA_CFG:?}" "${DATA_SRC:?}"

REPO=/home/ad158/le-wm
VENV=/scratch/ad158/le-wm-venv
OUT=/scratch/ad158/baselines/le-wm
NUM_WORKERS="${NUM_WORKERS:-12}"   # per rank

RUN_DIR="$OUT/runs/$RUN_NAME"
mkdir -p "$OUT"/{checkpoints,spt,hydra,logs,.cache/wandb} "$RUN_DIR"

export STABLEWM_HOME="$OUT"        # weights_epoch_*.pt -> $OUT/checkpoints/<RUN_NAME>/
export SPT_CACHE_DIR="$OUT/spt"    # spt.Manager reloads <run_dir>/checkpoints/last.ckpt on requeue
export WANDB_DIR="$RUN_DIR"
export WANDB_CACHE_DIR="$OUT/.cache/wandb"
export WANDB_INIT_TIMEOUT=300
export MUJOCO_GL=egl

echo "[job] $SLURM_JOB_ID on $(hostname) | run=$RUN_NAME | restart=${SLURM_RESTART_COUNT:-0} | $(date '+%F %T')"
nvidia-smi -L

# Stage the dataset in node-local RAM so 4 ranks x NUM_WORKERS don't hammer scratch.
SHM_DIR="/dev/shm/lewm-$SLURM_JOB_ID"
cleanup() { rm -rf "$SHM_DIR"; }
trap cleanup EXIT
mkdir -p "$SHM_DIR"
echo "[data] copying $DATA_SRC -> $SHM_DIR"
time cp "$DATA_SRC" "$SHM_DIR/"
DATA_PATH="$SHM_DIR/$(basename "$DATA_SRC")"

# --requeue only covers preemption / node failure; hitting --time is a TIMEOUT that is never
# requeued. Slurm sends USR1 to this shell 10 min before the wall (--signal=B:USR1@600) and we
# requeue ourselves. At most one epoch is lost: last.ckpt is rewritten every epoch.
requeue_self() {
  echo "[requeue] $(date '+%F %T') walltime near, requeueing $SLURM_JOB_ID"
  cleanup
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_self USR1

# Run from a per-run dir: spt and wandb write wandb_resume.json / wandb/ into the CWD, and a
# shared CWD would let one job's requeue resume another job's W&B run.
cd "$RUN_DIR"

# One srun task per GPU; Lightning's SLURM environment maps task -> device (devices=auto).
# Backgrounded + wait so the USR1 trap fires while training runs.
srun "$VENV/bin/python" "$REPO/train.py" \
  data="$DATA_CFG" \
  data.dataset.name="$DATA_PATH" \
  output_model_name="$RUN_NAME" \
  subdir="$RUN_NAME" \
  loader.num_workers="$NUM_WORKERS" \
  wandb.enabled=true \
  wandb.config.entity=ahitagnied-rice-university \
  wandb.config.project=baselines \
  wandb.config.name="$RUN_NAME" \
  wandb.config.id="$RUN_NAME-$SLURM_JOB_ID" \
  +wandb.config.group=le-wm \
  "hydra.run.dir=$OUT/hydra/$RUN_NAME/$SLURM_JOB_ID" \
  "$@" &
wait $! || exit $?
