"""360-degree orbit capture of the assembled DROID-ManipVerse scene.

Drives an external camera around a target point, saving compositied RGB
(splat BG + raytraced robot), per-frame metric depth, and surface normals.
No physics is ever stepped — it's a pure camera flythrough of a static scene.

Objects are reset to the scene's eval initial conditions (same as
``manipverse_random_action.py``); they are authored kinematic, so they hold
those poses with no settling. The Franka is posed at the episode's first
recorded joint config (``episode.pkl['traj']['joint_positions'][0]``) rather
than the upright default reset pose; pass ``--no-robot-pose`` to keep the
default, or ``--traj`` to point at a specific episode/trajectory pickle.

Companion to ``manipverse_random_action.py`` (same env, different driver).

Usage (on the SSH GPU box, after ``uv sync``):

    export MANIPVERSE_SCENE0_USD=/abs/path/to/scene_0/sim/scene.usda
    export POLARIS_ROBOT_SPLAT=0
    export POLARIS_RENDERER=gsplat
    # Optional: HDRI to relight the raytraced robot pixels
    export POLARIS_HDRI_PATH=/abs/path/to/some_workshop.hdr
    # Optional VRAM knobs (read in droid_cfg.py at env construction):
    export POLARIS_CAM_WIDTH=640 POLARIS_CAM_HEIGHT=360   # half-res
    export POLARIS_CAM_DEPTH=0 POLARIS_CAM_NORMALS=0      # drop G-buffers
    python experiments/manipverse_orbit_capture.py \
        --save-dir orbit_scene0 \
        --frames 240 --radius 0.9 --height 0.6 \
        --center 0.0 0.0 0.4 \
        --no-splat-depth                                  # skip 2nd splat pass

Outputs (under --save-dir):
    rgb/frame_XXXXXX.png       composited RGB (splat BG + raytraced robot)
    depth/frame_XXXXXX.png     uint16 millimetres, composited splat BG depth
                               + raytraced robot depth via semantic mask
    normals/frame_XXXXXX.png   surface normals encoded as (n*0.5+0.5)*255
                               (raytraced geometry only; splat BG = 0)
"""

import argparse
import os
import pickle
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from isaaclab.app import AppLauncher

# --- IsaacSim app must be launched before any IsaacLab-dependent imports. ---
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", default="orbit_scene0")
parser.add_argument("--external-cam", default="cam__22008760")
parser.add_argument("--env-id", default="DROID-ManipVerse-Scene0")
parser.add_argument("--frames", type=int, default=240, help="Total frames in the orbit.")
parser.add_argument("--radius", type=float, default=0.9, help="Final orbit radius (m).")
parser.add_argument("--height", type=float, default=0.6, help="Camera height above center (m).")
parser.add_argument(
    "--center",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.4],
    help="Look-at target xyz (world meters). Tune to your table center.",
)
parser.add_argument(
    "--zoom-in-frames",
    type=int,
    default=30,
    help="Frames to ease from radius*1.4 down to --radius at the start.",
)
parser.add_argument(
    "--no-splat-depth",
    action="store_true",
    help="Skip the second splat rasterization that produces BG depth. "
         "Halves splat memory + time per frame; depth/PNG falls back to "
         "raytraced (robot-only) sim depth.",
)
parser.add_argument(
    "--up-axis",
    choices=["z", "y"],
    default="z",
    help="World up axis (IsaacSim defaults to +Z).",
)
parser.add_argument(
    "--traj",
    default=None,
    help="Episode pickle holding the recorded joint stream used to pose the "
         "Franka (episode.pkl with ['traj']['joint_positions'], or a "
         "droid_trajectory.pkl with ['joint_positions']). Defaults to "
         "<env.usd_file>.parent/episode.pkl.",
)
parser.add_argument(
    "--no-robot-pose",
    dest="pose_robot",
    action="store_false",
    help="Skip snapping the Franka to the episode's initial joints; leave it "
         "at the default upright reset pose.",
)
parser.set_defaults(pose_robot=True)
parser.add_argument(
    "--gpu",
    default=None,
    help="Physical GPU index to pin the process to (sim + torch + gsplat). "
         "Defaults to CUDA_VISIBLE_DEVICES if set, else '0'. Pinning to one "
         "device avoids the multi-GPU cuda:0/cuda:N termination-manager crash.",
)
args_cli, _ = parser.parse_known_args()

# Pin to a single visible GPU BEFORE AppLauncher launches IsaacSim, so Kit /
# PhysX, torch, and gsplat all agree on one device (collapses to cuda:0).
if args_cli.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(
    f"[manipverse] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
    "(process pinned to a single GPU -> sim device cuda:0)",
    flush=True,
)

args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

import polaris.environments  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import (  # noqa: E402
    ManagerBasedRLSplatEnv,
)
from polaris.utils import load_eval_initial_conditions  # noqa: E402


PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _resolve_traj_path(env_usd_file, override):
    if override:
        return Path(override).expanduser().resolve()
    return Path(env_usd_file).parent / "episode.pkl"


def _load_initial_arm_q(path: Path) -> np.ndarray:
    """Return the episode's first recorded 7-DOF arm joint vector.

    Accepts either an episode.pkl (joints under ``["traj"]["joint_positions"]``)
    or a droid_trajectory.pkl (``["joint_positions"]``).
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "traj" in d and "joint_positions" in d["traj"]:
        qs = np.asarray(d["traj"]["joint_positions"], dtype=np.float64)
    elif isinstance(d, dict) and "joint_positions" in d:
        qs = np.asarray(d["joint_positions"], dtype=np.float64)
    else:
        raise KeyError(f"no joint_positions found in {path}")
    return qs[0]


def _snap_arm_to(env, q: np.ndarray) -> None:
    """Teleport the Franka's 7 arm joints to ``q`` (zero velocity).

    NOTE: orbit never steps physics, so this teleport is *not* driven back to
    the cfg default pose by the PD targets (a single sim.step would do exactly
    that, which is why orbit only renders, never steps).
    """
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    idx = [names.index(n) for n in PANDA_JOINT_NAMES]
    full_q = robot.data.joint_pos[0].detach().clone()
    full_q[idx] = torch.as_tensor(q, dtype=full_q.dtype, device=full_q.device)
    zero_v = torch.zeros_like(full_q)
    robot.write_joint_state_to_sim(
        full_q.unsqueeze(0), zero_v.unsqueeze(0),
        env_ids=torch.tensor([0], device=full_q.device),
    )


def look_at_quat_opengl(eye, target, up):
    """Return wxyz quaternion that orients an OpenGL camera (-Z forward, +Y up).

    IsaacLab's CameraCfg uses convention="opengl" for these scenes, so we
    build the same convention here.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)

    # OpenGL camera basis: x_cam=right, y_cam=up, z_cam=-forward (back).
    R = np.stack([r, u, -f], axis=1)

    t = R.trace()
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def save_rgb(rgb, path):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    iio.imwrite(path, rgb)


def save_depth(depth, path):
    """Encode metric depth as uint16 millimetres (range up to 65.5 m)."""
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        iio.imwrite(path, np.zeros(depth.shape, dtype=np.uint16))
        return
    d_mm = np.where(valid, np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)
    iio.imwrite(path, d_mm)


def composite_depth(splat_depth, sim_depth, sem_mask):
    """Replace splat depth with sim (raytraced) depth on robot/USD pixels.

    Mirrors ``ManagerBasedRLSplatEnv.custom_render`` RGB compositing: where
    the IsaacSim semantic mask says the pixel is raytraced geometry
    (``mask >= 2``) and the sim depth is finite & positive, prefer the sim
    depth; otherwise fall back to splat depth.
    """
    if isinstance(splat_depth, torch.Tensor):
        splat_depth = splat_depth.detach().cpu().numpy()
    if isinstance(sim_depth, torch.Tensor):
        sim_depth = sim_depth.detach().cpu().numpy()
    if isinstance(sem_mask, torch.Tensor):
        sem_mask = sem_mask.detach().cpu().numpy()
    splat_depth = np.asarray(splat_depth).squeeze().astype(np.float32)
    sim_depth = np.asarray(sim_depth).squeeze().astype(np.float32)
    sem_mask = np.asarray(sem_mask).squeeze()

    sim_valid = np.isfinite(sim_depth) & (sim_depth > 0)
    robot_pix = (sem_mask >= 2) & sim_valid
    return np.where(robot_pix, sim_depth, splat_depth)


def save_normals(normals, path):
    if isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().numpy()
    n = np.asarray(normals)
    if n.ndim == 4:
        n = n[0]
    if n.shape[-1] == 4:
        n = n[..., :3]
    n = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(path, n)


def main():
    save_dir = args_cli.save_dir
    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")
    normal_dir = os.path.join(save_dir, "normals")
    for d in (rgb_dir, depth_dir, normal_dir):
        os.makedirs(d, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.env_id,
        device="cuda:0",  # concrete index; process is pinned to one GPU above
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    # Match manipverse_random_action exactly: reset objects to the scene's eval
    # initial conditions. The objects are authored kinematic (see scene.usda:
    # physics:kinematicEnabled = 1), so they hold these poses regardless of
    # physics — there is nothing to "settle". random_action looks correct only
    # because it views from the original (forgiving) external-cam angle; the
    # residual base/table blending seen from grazing orbit angles is splat-on-
    # splat compositing, not a pose we can fix here.
    ic = {}
    try:
        language_instruction, initial_conditions = load_eval_initial_conditions(
            env.usd_file
        )
        print("Language instruction:", language_instruction)
        ic = initial_conditions[0]
    except Exception as e:
        print(f"[orbit] no eval initial conditions found ({e}); using defaults")
        ic = {}

    obs, info = env.reset(object_positions=ic)

    # Pose the Franka at the episode's first recorded joint config instead of
    # the upright default reset pose. Teleport only (write_joint_state_to_sim);
    # we must NOT step physics afterwards or the actuator PD targets (still the
    # cfg default pose) would immediately drag the arm back off q[0].
    if args_cli.pose_robot:
        traj_path = _resolve_traj_path(env.usd_file, args_cli.traj)
        try:
            q0 = _load_initial_arm_q(traj_path)
            _snap_arm_to(env, q0)
            env.sim.render()
            env.scene.update(0.0)
            print(f"[orbit] snapped Franka to episode q[0] from {traj_path}")
        except (FileNotFoundError, KeyError) as e:
            print(
                f"[orbit] could not pose robot from {traj_path} ({e}); "
                "leaving it at the default reset pose"
            )

    print("Available scene sensors:", list(env.scene.sensors.keys()))

    cam_name = args_cli.external_cam
    if cam_name not in env.scene.sensors:
        raise SystemExit(
            f"camera {cam_name!r} not found in env.scene.sensors — "
            f"pass --external-cam <name> from the list above."
        )

    base_cam = env.scene[cam_name]
    cam_data_types = set(base_cam.cfg.data_types)
    has_depth = "distance_to_image_plane" in cam_data_types
    has_normals = "normals" in cam_data_types
    if not (has_depth and has_normals):
        print(
            f"[orbit] note: camera {cam_name} data_types={sorted(cam_data_types)} — "
            f"depth/normals require updating droid_cfg.py."
        )

    center = np.asarray(args_cli.center, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0]) if args_cli.up_axis == "z" else np.array([0.0, 1.0, 0.0])

    T = args_cli.frames
    z_in = max(args_cli.zoom_in_frames, 0)

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    for i in range(T):
        if z_in and i < z_in:
            t = i / max(z_in - 1, 1)
            r = args_cli.radius * (1.4 - 0.4 * t)
        else:
            r = args_cli.radius

        a = 2.0 * np.pi * i / T
        eye = center + np.array(
            [r * np.cos(a), r * np.sin(a), args_cli.height], dtype=np.float64
        )
        quat_wxyz = look_at_quat_opengl(eye, center, up=up)

        pos_t = torch.tensor(eye, device=env.device, dtype=torch.float32).unsqueeze(0)
        quat_t = torch.tensor(quat_wxyz, device=env.device, dtype=torch.float32).unsqueeze(0)
        base_cam.set_world_poses(
            positions=pos_t,
            orientations=quat_t,
            env_ids=env_ids,
            convention="opengl",
        )

        # Force a sim render so camera output buffers and pos_w refresh.
        env.sim.render()
        env.scene.update(0)

        # transform_static=True so the splat renderer picks up the new
        # external-cam extrinsics this frame (wrist cams update every render
        # but external cams are otherwise treated as static).
        rgb_dict = env.custom_render(expensive=True, transform_static=True)
        rgb = rgb_dict.get(cam_name)
        if rgb is None:
            print(f"[orbit] frame {i:04d}: no RGB for {cam_name}, skipping.")
            continue
        save_rgb(rgb, os.path.join(rgb_dir, f"frame_{i:06d}.png"))

        out = base_cam.data.output

        # ----- Composited depth: splat BG + raytraced robot ----------------
        # `custom_render(transform_static=True)` already updated the splat
        # camera extrinsics this frame, so re-rendering rgbd here picks up
        # the same orbit pose. This is a second splat rasterization per
        # frame — fine for a flythrough, would be wasteful for an RL loop.
        # `--no-splat-depth` skips this entirely (saves the splat half of
        # the per-frame VRAM cost) and writes raytraced-only depth.
        if (
            not args_cli.no_splat_depth
            and has_depth
            and "distance_to_image_plane" in out
            and hasattr(env.splat_renderer, "render_rgbd")
        ):
            cam_pos = base_cam.data.pos_w[0].detach().cpu().numpy()
            cam_quat = base_cam.data.quat_w_world[0]
            cam_rot = math_utils.matrix_from_quat(cam_quat).detach().cpu().numpy()
            splat_out = env.splat_renderer.render_rgbd(
                {cam_name: {"pos": cam_pos, "rot": cam_rot}}
            )
            splat_depth = splat_out[cam_name]["depth"]

            sim_depth = out["distance_to_image_plane"][0]
            sem = out.get("semantic_segmentation")
            if sem is None:
                # Fallback: trust sim depth where it's finite, splat elsewhere.
                sem_mask = np.where(
                    np.isfinite(sim_depth.detach().cpu().numpy().squeeze()), 2, 0
                )
            else:
                sem_mask = sem[0]

            composed = composite_depth(splat_depth, sim_depth, sem_mask)
            save_depth(composed, os.path.join(depth_dir, f"frame_{i:06d}.png"))
        elif has_depth and "distance_to_image_plane" in out:
            # No splat depth available — write sim-only depth.
            save_depth(out["distance_to_image_plane"][0], os.path.join(depth_dir, f"frame_{i:06d}.png"))

        if has_normals and "normals" in out:
            save_normals(out["normals"][0], os.path.join(normal_dir, f"frame_{i:06d}.png"))

        if i % 20 == 0:
            print(f"[orbit] {i}/{T} eye={eye.round(3).tolist()}")

        # Splat tile-intersection allocates per-frame; on a tight 24 GB card
        # shared with other procs, fragmentation builds up fast.
        if torch.cuda.is_available() and (i % 8) == 0:
            torch.cuda.empty_cache()

    print(f"Saved {T} frames to {save_dir}/{{rgb,depth,normals}}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
