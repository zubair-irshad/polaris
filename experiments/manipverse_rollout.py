"""Replay manipverse cuRobo trajectories on a DROID-ManipVerse-* env.

Sibling to ``manipverse_random_action.py`` and ``manipverse_orbit_capture.py``
— same env loading, but instead of sampling random actions or orbiting an
external camera, this script reads a pickle of cuRobo-planned per-pick
joint trajectories produced by manipverse's pipeline/09_curobo_plan.py
and steps the env with those joint targets + a binary gripper command.

Trajectory pickle layout (matches manipverse.planning.curobo_runner.
PickPlaceTrajectory):

    {
        "<pick_name>": PickPlaceTrajectory(
            success: bool,
            joint_positions: np.ndarray (T, 7),    # absolute panda_joint*
            gripper_width:   np.ndarray (T,),      # metres (informational)
            attached:        np.ndarray (T,),      # 0=open, 1=closed → mapped
                                                   # to the env's binary
                                                   # finger action verbatim
            segment_boundaries: list[int],
            segment_names:      list[str],
            ...
        ),
        ...
    }

Default trajectory location is ``<env.usd_file>'s parent dir / trajectories.pkl``,
which is what scripts/register_polaris_env.py drops there when
--with-trajectories is set.

Usage:

    export POLARIS_ROBOT_SPLAT=0
    export POLARIS_RENDERER=gsplat
    python experiments/manipverse_rollout.py \
        --env-id DROID-ManipVerse-Scene5 \
        --save-dir camera_frames_droid_manipverse_scene5_rollout

Outputs:
    <save-dir>/<pick_idx>_<pick_name>/frame_XXXXXX.png   per-step composited RGB
    <save-dir>/<pick_idx>_<pick_name>.mp4                MP4 of the same
    <save-dir>/rollout_summary.json                       per-pick steps + status
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import imageio.v2 as imageio_v2
import imageio.v3 as iio
import numpy as np
import torch
from isaaclab.app import AppLauncher

# --- IsaacSim app must be launched before any IsaacLab-dependent imports. ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-dir",
    default="camera_frames_droid_manipverse_rollout",
    help="Where to write per-step PNGs + per-pick MP4s.",
)
parser.add_argument(
    "--external-cam",
    default="cam__22008760",
    help="Name of the external camera in scene.sensors.",
)
parser.add_argument(
    "--env-id",
    default="DROID-ManipVerse-Scene5",
)
parser.add_argument(
    "--traj",
    default=None,
    help="Path to manipverse trajectories.pkl. Defaults to "
         "<env.usd_file>'s parent / trajectories.pkl (i.e. the file "
         "register_polaris_env.py drops in the hub env dir).",
)
parser.add_argument(
    "--mp4-fps",
    type=float,
    default=30.0,
    help="Output MP4 frame rate. The trajectory's native step rate is "
         "interpolation_dt-defined by cuRobo (default 1/60 s).",
)
parser.add_argument(
    "--substeps",
    type=int,
    default=1,
    help="Replay 1 trajectory step per N env.step calls. Use >1 to slow "
         "the playback down for debugging.",
)
parser.add_argument(
    "--no-pngs", dest="save_pngs", action="store_false",
    help="Skip per-step PNG dumps; write only the per-pick MP4.",
)
parser.set_defaults(save_pngs=True)
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

import polaris.environments  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import (  # noqa: E402
    ManagerBasedRLSplatEnv,
)
from polaris.utils import load_eval_initial_conditions  # noqa: E402


# --------------------------------------------------------------------------- #
# Image helpers (copied from manipverse_random_action.py for parity)
# --------------------------------------------------------------------------- #

def save_rgb(rgb, path: str):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    iio.imwrite(path, rgb)


def combine_external_and_wrist(external, wrist):
    if external is None or wrist is None:
        return external if external is not None else wrist
    if external.shape[0] != wrist.shape[0]:
        from PIL import Image
        wrist_img = Image.fromarray(wrist)
        wrist = np.array(wrist_img.resize((wrist.shape[1], external.shape[0])))
    return np.concatenate((external, wrist), axis=1)


def _to_uint8(rgb):
    if rgb is None:
        return None
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# --------------------------------------------------------------------------- #
# Trajectory loading
# --------------------------------------------------------------------------- #

def _resolve_traj_path(env_usd_file: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(env_usd_file).parent / "trajectories.pkl"


def _load_trajectories(path: Path) -> dict:
    """Load and validate the manipverse trajectory pickle.

    The pickle stores a dict[name -> PickPlaceTrajectory]. We don't import
    the dataclass on the polaris side (manipverse isn't a dependency
    here), so each entry is treated as an attribute-bag (or dict).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"trajectories pickle not found: {path}\n"
            f"Run pipeline/09_curobo_plan.py and re-run scripts/register_polaris_env.py "
            f"with --with-trajectories on the manipverse side."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _attr(t, name, default=None):
    """Read an attribute or key off a trajectory (dataclass OR dict)."""
    if isinstance(t, dict):
        return t.get(name, default)
    return getattr(t, name, default)


# --------------------------------------------------------------------------- #
# Action assembly (PolaRiS DROID action space)
# --------------------------------------------------------------------------- #
# polaris.environments.droid_cfg.ActionCfg defines:
#   - arm:          JointPositionActionCfg(panda_joint.*)
#                   absolute targets, 7 dims, no offset
#   - finger_joint: BinaryJointPositionZeroToOneActionCfg
#                   1 dim, 0 = open, 1 = close (Robotiq mapping internal)
# Action space therefore is shape (8,) per env.
# --------------------------------------------------------------------------- #

def _action_for_step(joint_positions_t: np.ndarray,
                     attached_t: float,
                     device: str = "cuda") -> torch.Tensor:
    arm = torch.as_tensor(joint_positions_t, dtype=torch.float32, device=device)
    grip = torch.as_tensor([float(attached_t > 0.5)],
                           dtype=torch.float32, device=device)
    return torch.cat([arm, grip]).unsqueeze(0)        # (1, 8)


# --------------------------------------------------------------------------- #
# Per-pick playback
# --------------------------------------------------------------------------- #

def _replay_one(env: ManagerBasedRLSplatEnv,
                traj,
                ic: dict,
                save_subdir: Path,
                external_key: str,
                save_pngs: bool,
                substeps: int,
                mp4_fps: float) -> dict:
    save_subdir.mkdir(parents=True, exist_ok=True)
    name = _attr(traj, "name", "pick")

    # Reset env to IC every pick so they're independent (cuRobo planned
    # each one from the previous final joint state, but for visualisation
    # we treat each as a fresh demo).
    env.reset(object_positions=ic)

    qs = np.asarray(_attr(traj, "joint_positions"), dtype=np.float32)
    grip = np.asarray(_attr(traj, "attached"), dtype=np.float32)
    if qs is None or qs.size == 0:
        return {"name": name, "steps": 0, "frames": 0,
                "skipped_reason": "empty joint_positions"}

    n_steps = qs.shape[0]
    frames: list[np.ndarray] = []

    def grab_combined(obs_):
        splat = obs_.get("splat", {})
        ext = splat.get(external_key)
        wrist = splat.get("wrist_cam")
        return combine_external_and_wrist(_to_uint8(ext), _to_uint8(wrist))

    frame_id = 0
    for t in range(n_steps):
        action = _action_for_step(qs[t], float(grip[t]))
        for _ in range(max(1, int(substeps))):
            obs, rew, term, trunc, info = env.step(action, expensive=True)

        combined = grab_combined(obs)
        if combined is not None:
            if save_pngs:
                save_rgb(combined, str(save_subdir / f"frame_{frame_id:06d}.png"))
            frames.append(combined)
            frame_id += 1

        if bool(term[0]) or bool(trunc[0]):
            print(f"[rollout]   {name}: env terminated at step {t}/{n_steps}")
            break

    if frames:
        mp4_path = save_subdir.with_suffix(".mp4")
        imageio_v2.mimsave(str(mp4_path), frames, fps=mp4_fps)
        print(f"[rollout]   {name}: {len(frames)} frames -> {mp4_path.name}")

    rubric = info.get("rubric", {}) if "info" in dir() else {}
    return {
        "name":        name,
        "steps":       int(n_steps),
        "frames":      int(len(frames)),
        "success":     rubric.get("success") if rubric else None,
        "progress":    rubric.get("progress") if rubric else None,
    }


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.env_id,
        device="cuda:0",  # concrete index; process is pinned to one GPU above
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    # IC (mirrors manipverse_random_action.py)
    try:
        language_instruction, initial_conditions = load_eval_initial_conditions(
            env.usd_file
        )
        print(f"[rollout] instruction: {language_instruction!r}")
        ic = initial_conditions[0] if initial_conditions else {}
    except Exception as e:
        print(f"[rollout] no eval ICs ({e}); using defaults")
        ic = {}

    print(f"[rollout] scene sensors: {list(env.scene.sensors.keys())}")

    # Trajectories
    traj_path = _resolve_traj_path(env.usd_file, args_cli.traj)
    print(f"[rollout] trajectories: {traj_path}")
    trajectories = _load_trajectories(traj_path)

    successful = {n: t for n, t in trajectories.items()
                  if _attr(t, "success", False)}
    print(f"[rollout] replaying {len(successful)}/{len(trajectories)} picks")

    summary = []
    for i, (name, traj) in enumerate(successful.items()):
        print(f"\n[rollout] {i+1}/{len(successful)}  {name}")
        safe = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        sub = save_dir / f"{i:02d}_{safe}"
        info = _replay_one(env, traj, ic, sub,
                           external_key=args_cli.external_cam,
                           save_pngs=args_cli.save_pngs,
                           substeps=args_cli.substeps,
                           mp4_fps=args_cli.mp4_fps)
        summary.append(info)

    out_summary = save_dir / "rollout_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n[rollout] summary -> {out_summary}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
