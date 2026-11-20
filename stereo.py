"""Stereo rig for the OGBench Cube env: the dataset's camera plus a copy shifted along its x axis.

The cameras are passed to stable-worldmodel's ``CubeEnv(cameras=...)``, which renders them as
``pixels.left`` / ``pixels.right``. The same function builds the rig for the dataset
(``render_stereo.py``) and for the online evaluation (``eval.py``).
"""

import numpy as np

LEFT, RIGHT = "left", "right"
PIXEL_KEYS = (f"pixels.{LEFT}", f"pixels.{RIGHT}")


def stereo_cameras(baseline: float = 0.06) -> list[dict]:
    """Camera configurations of the rig: ``left`` is the env's default view (``front_pixels``), ``right`` has the
    same orientation and sits ``baseline`` metres along the left camera's x axis."""
    from stable_worldmodel.envs.ogbench.cube_env import DEFAULT_CAMERAS

    front = DEFAULT_CAMERAS["front_pixels"]
    x_axis = np.asarray(front["xyaxes"][:3], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    right_pos = np.asarray(front["pos"], dtype=np.float64) + float(baseline) * x_axis
    return [
        {"name": LEFT, "pos": tuple(front["pos"]), "xyaxes": tuple(front["xyaxes"])},
        {"name": RIGHT, "pos": tuple(right_pos.tolist()), "xyaxes": tuple(front["xyaxes"])},
    ]


def disparity_report(env, baseline: float, height: float = 0.02) -> dict:
    """Stereo geometry of the rig on a reset ``CubeEnv``: focal length in pixels and, at the near / centre / far
    end of the cube workspace (the object sampling bounds at ``height`` above the table), the depth along the
    optical axis, the disparity in pixels and the pixels of disparity per cm of depth."""
    import mujoco

    env = env.unwrapped
    model, data = env._model, env._data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, LEFT)
    image_height = int(env._render_height)
    focal = (image_height / 2) / np.tan(np.deg2rad(float(model.cam_fovy[cam])) / 2)
    pos, rot = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)

    (x0, y0), (x1, y1) = np.asarray(env._object_sampling_bounds, dtype=np.float64)
    points = [(x, y, height) for x in (x0, x1) for y in (y0, y1)]
    centre = ((x0 + x1) / 2, (y0 + y1) / 2, height)
    # a MuJoCo camera looks along its -z axis
    depth = lambda p: float(-(rot.T @ (np.asarray(p) - pos))[2])
    depths = {"near": min(map(depth, points)), "centre": depth(centre), "far": max(map(depth, points))}
    rows = {
        name: dict(depth_m=z, disparity_px=focal * baseline / z, px_per_cm_depth=focal * baseline / z**2 * 0.01)
        for name, z in depths.items()
    }
    return dict(
        baseline_m=float(baseline),
        image_height_px=image_height,
        fovy_deg=float(model.cam_fovy[cam]),
        focal_px=float(focal),
        workspace=rows,
    )


def format_report(report: dict) -> str:
    lines = [
        (
            f"baseline {report['baseline_m'] * 100:.1f} cm | fovy {report['fovy_deg']:.1f} deg | "
            f"focal {report['focal_px']:.1f} px at {report['image_height_px']} px"
        ),
        f"{'':8}{'depth (m)':>10}{'disparity (px)':>16}{'px per cm depth':>17}",
    ]
    for name, row in report["workspace"].items():
        lines.append(f"{name:8}{row['depth_m']:>10.3f}{row['disparity_px']:>16.2f}{row['px_per_cm_depth']:>17.3f}")
    return "\n".join(lines)
