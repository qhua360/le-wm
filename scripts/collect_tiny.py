"""Collect a tiny mono Cube-Single dataset (h5) with the expert, for smoke tests only.

    python scripts/collect_tiny.py --out $STABLEWM_HOME/ogbench/cube_tiny.h5 --episodes 4 --steps 60
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


def collect(out, episodes=4, steps=60, seed=0, num_envs=1):
    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=num_envs,
        image_shape=(224, 224),
        max_episode_steps=steps,
        env_type="single",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )
    world.set_policy(ExpertPolicy())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=episodes, seed=seed, format="hdf5", progress=False)
    world.close()
    # info keys such as privileged/block_0_pos become h5 groups; the released datasets have flat column names
    with h5py.File(out, "r+") as f:
        nested = []
        f.visititems(lambda name, item: nested.append(name) if isinstance(item, h5py.Dataset) and "/" in name else None)
        for name in nested:
            f.move(name, name.replace("/", "_"))
        for name in [k for k, v in f.items() if isinstance(v, h5py.Group)]:
            del f[name]
        if "ep_idx" not in f:  # per-row episode index, which eval.py reads (present in the released datasets)
            f["ep_idx"] = np.repeat(np.arange(len(f["ep_len"])), f["ep_len"][:])
    return Path(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(collect(args.out, args.episodes, args.steps, args.seed))
