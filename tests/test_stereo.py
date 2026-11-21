"""CPU tests of the stereo LeWM pieces: model input, camera rig, dataset rendering."""

import os

os.environ.setdefault("MUJOCO_GL", "glfw" if os.uname().sysname == "Darwin" else "egl")

import h5py
import hydra
import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import render_stereo
from stereo import LEFT, PIXEL_KEYS, RIGHT, disparity_report, stereo_cameras
from utils import get_view_preprocessor

IMG = 56


def build(model_cfg):
    torch.manual_seed(0)
    cfg = OmegaConf.create({"img_size": IMG, "history_size": 3, "embed_dim": 192})
    cfg.model = OmegaConf.load(f"config/train/model/{model_cfg}.yaml")
    cfg.model.action_encoder.input_dim = 10
    return hydra.utils.instantiate(cfg.model).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def mono():
    return build("lewm")


@pytest.fixture(scope="module")
def stereo():
    return build("lewm_stereo")


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_stereo_differs_only_in_patch_embedding(mono, stereo):
    a, b = mono.state_dict(), stereo.state_dict()
    assert a.keys() == b.keys()
    different = [k for k in a if a[k].shape != b[k].shape]
    assert len(different) == 1 and "patch_embeddings" in different[0]
    assert b[different[0]].shape[1] == 2 * a[different[0]].shape[1] == 6
    assert mono.pixel_keys == ("pixels",) and stereo.pixel_keys == PIXEL_KEYS
    print(f"encoder parameters: mono {count(mono.encoder):,}, stereo {count(stereo.encoder):,}")
    assert count(stereo.encoder) - count(mono.encoder) == 3 * 14 * 14 * 192


def test_encode_stacks_views(stereo):
    batch = {k: torch.randn(2, 4, 3, IMG, IMG) for k in PIXEL_KEYS}
    batch["action"] = torch.randn(2, 4, 10)
    out = stereo.encode(dict(batch))
    assert out["emb"].shape == (2, 4, 192)
    swapped = stereo.encode({**batch, PIXEL_KEYS[0]: batch[PIXEL_KEYS[1]], PIXEL_KEYS[1]: batch[PIXEL_KEYS[0]]})
    assert not torch.allclose(out["emb"], swapped["emb"])  # both views are read, in order


@pytest.mark.parametrize("name", ["mono", "stereo"])
def test_get_cost_shapes(name, request):
    model = request.getfixturevalue(name)
    B, S, T = 1, 3, 5  # envs per solver batch (cem.yaml: batch_size 1), action candidates, horizon
    info = {"action": torch.zeros(B, S, 1, 10)}
    for key in model.pixel_keys:
        info[key] = torch.randn(B, S, 1, 3, IMG, IMG)
        info["goal" if key == "pixels" else f"goal_{key}"] = torch.randn(B, S, 1, 3, IMG, IMG)
    cost = model.get_cost(info, torch.randn(B, S, T, 10))
    assert cost.shape == (B, S) and torch.isfinite(cost).all()


def test_view_preprocessor_handles_dotted_keys():
    frames = torch.randint(0, 256, (4, 3, IMG, IMG), dtype=torch.uint8)
    plain = get_view_preprocessor("pixels", IMG)({"pixels": frames.clone()})["pixels"]
    dotted = get_view_preprocessor(PIXEL_KEYS[0], IMG)({PIXEL_KEYS[0]: frames.clone()})
    assert set(dotted) == {PIXEL_KEYS[0]}
    torch.testing.assert_close(dotted[PIXEL_KEYS[0]], plain)


def test_rig_geometry():
    baseline = 0.06
    env = render_stereo.make_env(baseline)
    model, data = env._model, env._data
    left, right = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in (LEFT, RIGHT))
    x_axis = data.cam_xmat[left].reshape(3, 3)[:, 0]
    np.testing.assert_allclose(data.cam_xpos[right] - data.cam_xpos[left], baseline * x_axis, atol=1e-9)
    np.testing.assert_allclose(data.cam_xmat[right], data.cam_xmat[left], atol=1e-9)
    assert model.cam_fovy[right] == model.cam_fovy[left]

    report = disparity_report(env, baseline)
    rows = report["workspace"]
    assert rows["near"]["depth_m"] < rows["centre"]["depth_m"] < rows["far"]["depth_m"]
    assert rows["near"]["disparity_px"] > rows["far"]["disparity_px"] > 0
    np.testing.assert_allclose(report["focal_px"], 112 / np.tan(np.deg2rad(report["fovy_deg"]) / 2))
    env.close()


def test_zero_baseline_reproduces_left_view():
    env = render_stereo.make_env(0.0)
    left, right = render_stereo.render_pair(env, env._data.qpos.copy(), env._data.qvel.copy())
    assert np.abs(left.astype(int) - right.astype(int)).max() <= 1
    env.close()
    assert stereo_cameras(0.06)[0]["pos"] == stereo_cameras(0.2)[0]["pos"]


@pytest.fixture(scope="module")
def tiny_h5(tmp_path_factory):
    from scripts.collect_tiny import collect

    return collect(tmp_path_factory.mktemp("data") / "cube_tiny.h5", episodes=2, steps=12)


def test_render_stereo_writes_both_views(tiny_h5):
    dst = tiny_h5.with_name("cube_tiny_stereo.h5")
    args = ["--src", str(tiny_h5), "--dst", str(dst), "--check-rows", "10", "--workers", "1", "--batch", "8"]
    assert render_stereo.main(args) == 0
    with h5py.File(tiny_h5) as src, h5py.File(dst) as out:
        left, right = (out[k] for k in PIXEL_KEYS)
        assert left.shape == right.shape == src["pixels"].shape and right.dtype == np.uint8
        np.testing.assert_array_equal(left[:], src["pixels"][:])
        assert right.attrs["rows_done"] == len(right)
        assert all(np.abs(right[i].astype(int) - left[i]).mean() > 1 for i in (0, len(right) - 1))
        np.testing.assert_array_equal(out["qpos"][:], src["qpos"][:])
    assert render_stereo.main(args) == 0  # resuming a finished file is a no-op

    truncated = tiny_h5.with_name("cube_tiny_one_episode.h5")
    assert render_stereo.main([*args[:3], str(truncated), *args[4:], "--episodes", "1"]) == 0
    with h5py.File(tiny_h5) as src, h5py.File(truncated) as out:
        rows = int(src["ep_len"][0])
        assert len(out["ep_len"]) == 1 and len(out["pixels.right"]) == len(out["action"]) == rows


def test_left_check_rejects_wrong_states(tiny_h5, tmp_path):
    broken = tmp_path / "broken.h5"
    with h5py.File(tiny_h5) as src, h5py.File(broken, "w") as out:
        for name in src:
            src.copy(name, out)
        out["pixels"][:] = 255 - out["pixels"][:]
    with pytest.raises(SystemExit, match="not reproduced"):
        render_stereo.main(["--src", str(broken), "--check-only", "--check-rows", "5"])


def test_plugin_compression_is_kept(tiny_h5, tmp_path):
    hdf5plugin = pytest.importorskip("hdf5plugin")
    blosc = tmp_path / "blosc.h5"
    with h5py.File(tiny_h5) as src, h5py.File(blosc, "w") as out:
        for name in src:
            if name == "pixels":
                out.create_dataset(name, data=src[name][:], chunks=(4, 224, 224, 3), **hdf5plugin.Blosc(cname="lz4"))
            else:
                src.copy(name, out)
    dst = tmp_path / "blosc_stereo.h5"
    args = ["--src", str(blosc), "--dst", str(dst), "--check-rows", "5", "--workers", "1", "--episodes", "1"]
    assert render_stereo.main(args) == 0
    with h5py.File(blosc) as src, h5py.File(dst) as out:
        rows = len(out["pixels"])
        np.testing.assert_array_equal(out["pixels"][:], src["pixels"][:rows])
        for key in ("pixels", "pixels.right"):
            filters = out[key].id.get_create_plist()
            assert filters.get_filter(0)[0] == 32001 and out[key].chunks == (4, 224, 224, 3)
        assert np.abs(out["pixels.right"][0].astype(int) - out["pixels"][0]).mean() > 1
