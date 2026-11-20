#!/usr/bin/env bash
# End-to-end smoke of mono and stereo LeWM on Cube-Single: dataset -> render_stereo.py -> a short train.py run ->
# eval.py on a few episodes. Checks that the code runs; the numbers mean nothing.
#
#   scripts/smoke_stereo.sh                       # tiny dataset collected here (CPU or GPU)
#   SRC=/path/cube_single_expert.h5 EPISODES=200 scripts/smoke_stereo.sh    # first episodes of a real dataset
#
# Env: DEVICE=cuda|cpu (default: cuda if available), BASELINE (0.06), STABLEWM_HOME (default: a temp dir),
#      TRAIN_BATCHES (20), EVAL_EPISODES (2), PYTHON (.venv/bin/python), FULL_CEM=1 keeps the real CEM settings.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PY="${PYTHON:-$REPO/.venv/bin/python}"
DEVICE="${DEVICE:-$($PY -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"
BASELINE="${BASELINE:-0.06}"
TRAIN_BATCHES="${TRAIN_BATCHES:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
export STABLEWM_HOME="${STABLEWM_HOME:-$(mktemp -d)}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-$STABLEWM_HOME/spt}"
if [[ -z "${MUJOCO_GL:-}" ]]; then
    [[ "$(uname)" == Darwin ]] && export MUJOCO_GL=glfw || export MUJOCO_GL=egl
fi
DATA="$STABLEWM_HOME/datasets/ogbench"
MONO="$DATA/smoke_mono.h5"
STEREO="$DATA/smoke_stereo.h5"
mkdir -p "$DATA" "$STABLEWM_HOME/smoke_run"
echo "== smoke: device=$DEVICE baseline=$BASELINE MUJOCO_GL=$MUJOCO_GL home=$STABLEWM_HOME"

echo "== dataset"
if [[ -n "${SRC:-}" ]]; then
    "$PY" render_stereo.py --src "$SRC" --check-only --out "$STABLEWM_HOME/left_check_full"
    "$PY" render_stereo.py --src "$SRC" --dst "$STEREO" --baseline "$BASELINE" --episodes "${EPISODES:-200}" \
        --out "$STABLEWM_HOME/stereo_check"
else
    [[ -e "$MONO" ]] || "$PY" scripts/collect_tiny.py --out "$MONO" --episodes 6 --steps 60
    "$PY" render_stereo.py --src "$MONO" --dst "$STEREO" --baseline "$BASELINE" --check-rows 100 \
        --out "$STABLEWM_HOME/stereo_check"
fi
# the stereo file keeps `pixels`, so the mono run reads the same rows from it

if [[ "$DEVICE" == cuda ]]; then
    TRAIN_DEVICE=(trainer.accelerator=gpu trainer.devices=1)  # config defaults otherwise: bf16, batch 128
    WORKERS=(num_workers=6)
else
    TRAIN_DEVICE=(trainer.accelerator=cpu trainer.precision=32 loader.batch_size=8)
    WORKERS=(num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null)
fi
if [[ -n "${FULL_CEM:-}" ]]; then
    CEM=()
else
    CEM=(solver.num_samples=16 solver.n_steps=3 solver.topk=4)
fi
# short episodes of the tiny dataset need a short goal offset
EVAL=(eval.num_eval="$EVAL_EPISODES" solver.device="$DEVICE" +device="$DEVICE" "${CEM[@]}")
[[ -n "${SRC:-}" ]] || EVAL+=(eval.goal_offset_steps=10 eval.eval_budget=10 plan_config.horizon=2 plan_config.receding_horizon=2)

cd "$STABLEWM_HOME/smoke_run"
for arm in mono stereo; do
    if [[ $arm == stereo ]]; then
        TRAIN=(data=ogb_stereo model=lewm_stereo); EVAL_CFG=cube_stereo
    else
        TRAIN=(data=ogb); EVAL_CFG=cube
    fi
    echo "== train $arm ($TRAIN_BATCHES batches)"
    "$PY" "$REPO/train.py" "${TRAIN[@]}" data.dataset.name=ogbench/smoke_stereo.h5 \
        output_model_name="smoke_$arm" subdir="smoke_$arm" trainer.max_epochs=1 \
        +trainer.limit_train_batches="$TRAIN_BATCHES" +trainer.limit_val_batches=2 \
        "${TRAIN_DEVICE[@]}" "${WORKERS[@]}"
    echo "== eval $arm ($EVAL_EPISODES episodes)"
    "$PY" "$REPO/eval.py" --config-name="$EVAL_CFG" policy="smoke_$arm/weights_epoch_1.pt" \
        eval.dataset_name=ogbench/smoke_stereo "${EVAL[@]}"
done
echo "== smoke passed: mono and stereo trained and evaluated ($STABLEWM_HOME)"
