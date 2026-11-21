#!/usr/bin/env bash
# One-box setup on a fresh Lambda instance (Lambda Stack 24.04): toolchain, venv, tests, and LeWM's released
# Cube-Single dataset from the Hugging Face Hub. Run from the repo root. stable-worldmodel is installed from a
# checkout next to this repo (../stable-worldmodel, the branch with CubeEnv(cameras=...)) or from SWM_SPEC.
# Needs ~160 GB free for the dataset (46 GB download + the extracted h5); DATASET=0 skips it.
set -euo pipefail
cd "$(dirname "$0")/.."
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL=egl
SWM_DIR="${SWM_DIR:-$(cd .. && pwd)/stable-worldmodel}"
SWM_SPEC="${SWM_SPEC:--e $SWM_DIR[train,env,format]}"
SPT_REV=ab836bf699a2be712dfcf980a5eb70d36391e876   # stable-pretraining revision the code was tested with
H5="$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5"
mkdir -p "$STABLEWM_HOME/datasets/ogbench" "$STABLEWM_HOME/checkpoints"

echo "== apt"
sudo apt-get update -qq >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd swig build-essential libgl1 libegl1 >/dev/null
# Lambda Stack ships the compute libraries only; without the NVIDIA GL libraries EGL falls back to Mesa's software
# renderer (llvmpipe), ~150x slower. Install the ones matching the loaded driver.
if ! ldconfig -p | grep -q libEGL_nvidia; then
    PKG="$(dpkg-query -W -f='${Package}\n' 'libnvidia-compute-*' | head -1)"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        "${PKG/compute/gl}=$(dpkg-query -W -f='${Version}' "$PKG")" >/dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

echo "== venv"
[[ -d .venv ]] || uv venv --python 3.11 .venv
# shellcheck disable=SC2086
uv pip install -q --python .venv/bin/python $SWM_SPEC \
    "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@$SPT_REV" pytest huggingface_hub
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), f'torch {torch.__version__} (cuda {torch.version.cuda}) cannot init on this driver'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
.venv/bin/python -c "
import mujoco
renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8)  # keeps the context current
renderer.render()
from OpenGL import GL
try: name = GL.glGetString(GL.GL_RENDERER)
except Exception as error: name = getattr(error, 'result', b'')
print('mujoco', mujoco.__version__, 'EGL renderer', name)
assert name and b'llvmpipe' not in name, 'EGL is rendering in software (llvmpipe), not on the GPU'
"

echo "== tests"
.venv/bin/python -m pytest tests/test_stereo.py -q
if [[ -d "$SWM_DIR/tests" ]]; then
    (cd "$SWM_DIR" && "$OLDPWD/.venv/bin/python" -m pytest tests/test_new_world.py -q)
fi

if [[ "${DATASET:-1}" == 1 && ! -e "$H5" ]]; then
    echo "== dataset (quentinll/lewm-cube, 46 GB)"
    DL="$STABLEWM_HOME/downloads/lewm-cube"
    .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('quentinll/lewm-cube', repo_type='dataset', local_dir='$DL')"
    tar --use-compress-program=unzstd -xf "$DL/cube_single_expert.tar.zst" -C "$DL"
    FOUND="$(find "$DL" -name 'cube_single_expert*.h5' | head -1)"
    [[ -n "$FOUND" ]] || { echo "no cube_single_expert h5 in the archive:"; find "$DL" -maxdepth 3; exit 1; }
    mv "$FOUND" "$H5"
    rm -f "$DL/cube_single_expert.tar.zst"
fi
ls -la "$STABLEWM_HOME/datasets/ogbench"
echo "== setup done"
