"""Add a stereo pair to a collected OGBench Cube dataset by restoring each stored simulator state.

    python render_stereo.py --src cube_single_expert.h5 --dst cube_single_expert_stereo_b06.h5 --baseline 0.06

1. Left check: random rows are re-rendered from their stored ``qpos`` / ``qvel`` with the left camera and compared
   with the stored ``pixels``. Above ``--tolerance`` the script stops: the dataset cannot be re-rendered faithfully.
2. Prints the stereo geometry (focal length, disparity over the workspace) and saves a few left / right pairs.
3. Writes ``dst``: a copy of ``src`` with ``pixels.left`` (an HDF5 link to ``pixels``, no extra storage) and
   ``pixels.right``. Rendering can be interrupted and resumed with the same command.

``--check-only`` stops after step 2; ``--episodes N`` keeps only the first N episodes (smoke tests).
"""

import argparse
import json
import os
import shutil
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

from stereo import LEFT, PIXEL_KEYS, RIGHT, disparity_report, format_report, stereo_cameras

INDEX_COLS = ("ep_len", "ep_offset")
_ENV = None  # one env per worker process


def datasets(handle):
    """Every dataset of an h5 file by its path (columns may sit in groups, e.g. ``privileged/block_0_pos``)."""
    found = {}
    handle.visititems(lambda name, item: found.update({name: item}) if isinstance(item, h5py.Dataset) else None)
    return found


def make_env(baseline, env_type="single", size=224):
    """The env of config/eval/cube.yaml with the stereo cameras."""
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    env = CubeEnv(
        env_type=env_type,
        ob_type="states",
        width=size,
        height=size,
        visualize_info=False,
        cameras=stereo_cameras(baseline),
    )
    env.reset(seed=0)
    return env


def render_pair(env, qpos, qvel):
    env.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    views = env.render_multiview()
    return views[LEFT], views[RIGHT]


def _init_worker(baseline, env_type, size):
    global _ENV
    _ENV = make_env(baseline, env_type, size)


def _render_right_rows(args):
    start, qpos, qvel = args
    return start, np.stack([render_pair(_ENV, q, v)[1] for q, v in zip(qpos, qvel)])


def left_check(src, env, rows, seed):
    """Mean / max absolute difference (0-255) between the stored ``pixels`` and the re-rendered left view."""
    total = src["pixels"].shape[0]
    picks = np.sort(np.random.default_rng(seed).choice(total, size=min(rows, total), replace=False))
    per_row, worst, exact, examples = [], 0, 0, []
    for row in picks:
        stored = src["pixels"][row]
        left, right = render_pair(env, src["qpos"][row], src["qvel"][row])
        diff = np.abs(stored.astype(np.int16) - left.astype(np.int16))
        per_row.append(float(diff.mean()))
        worst = max(worst, int(diff.max()))
        exact += int(diff.max() == 0)
        if len(examples) < 64:
            examples.append((int(row), stored, left, right))
    per_row = np.asarray(per_row)
    stats = dict(
        rows=int(len(picks)),
        mean_abs_diff=float(per_row.mean()),
        worst_row_mean_abs_diff=float(per_row.max()),
        max_abs_diff=worst,
        exact_rows=exact,
    )
    return stats, examples


def save_pairs(examples, out_dir, count):
    """PNG panels: stored pixels | re-rendered left | right."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for row, stored, left, right in examples[:count]:
        Image.fromarray(np.concatenate([stored, left, right], axis=1)).save(out_dir / f"row_{row:08d}.png")


def create_dst(src_path, dst_path, episodes):
    """``dst`` as a full copy of ``src``, or its first ``episodes`` episodes."""
    if dst_path.exists():
        return
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if episodes is None:
        shutil.copyfile(src_path, tmp)
    else:
        with h5py.File(src_path, "r") as src, h5py.File(tmp, "w") as dst:
            lengths = src["ep_len"][:episodes]
            rows = int(src["ep_offset"][episodes - 1] + lengths[-1])
            for name, column in datasets(src).items():
                count = episodes if name in INDEX_COLS else rows
                kwargs = dict(compression=column.compression, compression_opts=column.compression_opts)
                if column.chunks:
                    kwargs["chunks"] = (min(column.chunks[0], count), *column.chunks[1:])
                out = dst.create_dataset(name, shape=(count, *column.shape[1:]), dtype=column.dtype, **kwargs)
                for start in range(0, count, 4096):
                    out[start : start + 4096] = column[start : min(start + 4096, count)]
    tmp.replace(dst_path)


def render_right(dst_path, baseline, env_type, workers, batch):
    """Fill ``pixels.right`` of ``dst`` (resumable: the ``rows_done`` attribute)."""
    left_key, right_key = PIXEL_KEYS
    with h5py.File(dst_path, "r+") as dst:
        pixels = dst["pixels"]
        total = pixels.shape[0]
        if left_key not in dst:
            dst[left_key] = pixels  # hard link: the same data under the left view's name
        if right_key not in dst:
            kwargs = dict(compression=pixels.compression, compression_opts=pixels.compression_opts)
            if pixels.chunks:
                kwargs["chunks"] = pixels.chunks
            column = dst.create_dataset(right_key, shape=pixels.shape, dtype=np.uint8, **kwargs)
            column.attrs["rows_done"] = 0
            column.attrs["baseline_m"] = float(baseline)
        column = dst[right_key]
        if float(column.attrs["baseline_m"]) != float(baseline):
            raise ValueError(f"{dst_path} holds a {column.attrs['baseline_m']} m right view, not {baseline} m")
        done = int(column.attrs["rows_done"])
        qpos, qvel = dst["qpos"], dst["qvel"]
        jobs = ((s, qpos[s : s + batch], qvel[s : s + batch]) for s in range(done, total, batch))
        size = int(pixels.shape[1])
        with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(baseline, env_type, size)) as pool:
            for start, frames in pool.imap(_render_right_rows, jobs):  # ordered, so rows_done is a prefix
                column[start : start + len(frames)] = frames
                column.attrs["rows_done"] = start + len(frames)
                if (start // batch) % 50 == 0:
                    dst.flush()
                    print(f"rendered {start + len(frames)} / {total} rows", flush=True)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="collected dataset (.h5) with pixels, qpos, qvel")
    parser.add_argument("--dst", type=Path, help="output dataset (.h5); required unless --check-only")
    parser.add_argument("--baseline", type=float, default=0.06, help="metres between the two cameras")
    parser.add_argument("--env-type", default="single")
    parser.add_argument("--check-rows", type=int, default=1000, help="rows of the left check")
    parser.add_argument("--tolerance", type=float, default=2.0, help="largest accepted mean abs difference (0-255)")
    parser.add_argument("--pairs", type=int, default=8, help="stored | left | right panels to save")
    parser.add_argument("--out", type=Path, help="folder for the report and panels (default: next to --dst / --src)")
    parser.add_argument("--episodes", type=int, help="keep only the first N episodes in --dst")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch", type=int, default=256, help="rows per render job")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and args.dst is None:
        parser.error("--dst is required unless --check-only")
    if args.dst is not None and args.dst.resolve() == args.src.resolve():
        parser.error("--dst must differ from --src")
    out = args.out or (args.dst or args.src).with_suffix("").with_name((args.dst or args.src).stem + "_stereo_check")

    with h5py.File(args.src, "r") as src:
        columns = {name: (tuple(col.shape), str(col.dtype)) for name, col in datasets(src).items()}
        print("columns:")
        for name, (shape, dtype) in sorted(columns.items()):
            print(f"  {name:40} {shape} {dtype}")
        missing = [name for name in ("pixels", "qpos", "qvel", *INDEX_COLS) if name not in columns]
        if missing:
            raise SystemExit(f"{args.src}: missing columns {missing}; the states cannot be restored")
        size = columns["pixels"][0][1]
        env = make_env(args.baseline, args.env_type, size)
        if columns["qpos"][0][1:] != (env._model.nq,) or columns["qvel"][0][1:] != (env._model.nv,):
            raise SystemExit(f"{args.src}: qpos / qvel do not fit the {args.env_type} cube env")
        stats, examples = left_check(src, env, args.check_rows, args.seed)
    report = disparity_report(env, args.baseline)
    env.close()

    out.mkdir(parents=True, exist_ok=True)
    save_pairs(examples, out, args.pairs)
    passed = stats["mean_abs_diff"] <= args.tolerance
    (out / "report.json").write_text(
        json.dumps(dict(left_check=dict(stats, tolerance=args.tolerance, passed=passed), stereo=report), indent=2)
    )
    print(f"left check on {stats['rows']} rows: mean abs diff {stats['mean_abs_diff']:.3f}, worst row "
          f"{stats['worst_row_mean_abs_diff']:.3f}, max {stats['max_abs_diff']}, exact rows {stats['exact_rows']} "
          f"(tolerance {args.tolerance}) -> {'ok' if passed else 'FAILED'}")  # fmt: skip
    print(format_report(report))
    print(f"panels (stored | left | right) and report.json: {out}")
    if not passed:
        raise SystemExit("the stored pixels are not reproduced from qpos / qvel; not rendering the right view")
    if args.check_only:
        return 0

    create_dst(args.src, args.dst, args.episodes)
    rows = render_right(args.dst, args.baseline, args.env_type, args.workers, args.batch)
    print(f"{args.dst}: {PIXEL_KEYS[0]} (link to pixels) and {PIXEL_KEYS[1]} for {rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
