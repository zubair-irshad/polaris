"""Replay the *real* DROID trajectory on a DROID-ManipVerse-* env.

Companion to ``manipverse_random_action.py`` (random action smoke test)
and ``manipverse_rollout.py`` (cuRobo plan playback). This one is the
"is the real2sim reconstruction faithful?" probe: it streams the
recorded joint positions from the DROID episode back into the IsaacSim
Franka as absolute joint targets, so a perfectly reconstructed scene
should reproduce the real video.

The trajectory pickle is produced by ``pipeline/11_droid_replay.py``
(``droid_trajectory.pkl``) and dropped next to ``scene.usda`` by
``scripts/register_polaris_env.py`` (mirroring how stage 09 ships
``trajectories.pkl``). Layout::

    {
      "name":               str,
      "joint_positions":    np.ndarray (T, 7),
      "gripper_position":   np.ndarray (T,)   in [0, 1]
      "gripper_binary":     np.ndarray (T,)   {0, 1}
      "cartesian_position": np.ndarray (T, 6) base-frame EE (diagnostics)
      "dt":                 float             (≈ 1/15 s for DROID)
      "episode_id":         str
      "raw_actions":        dict              full DROID action dict
    }

Usage::

    export POLARIS_ROBOT_SPLAT=0
    export POLARIS_RENDERER=gsplat
    python experiments/manipverse_droid_replay.py \
        --env-id DROID-ManipVerse-Scene5

Outputs:
    <save-dir>/frame_XXXXXX.png             per-step composite RGB
    <save-dir>/droid_replay.mp4             MP4 of the same
    <save-dir>/droid_replay_summary.json    per-step real vs sim joint MSE
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio_v2
import imageio.v3 as iio
import numpy as np
import torch
from isaaclab.app import AppLauncher

# --- IsaacSim app must be launched before any IsaacLab-dependent imports. ---
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir",
                    default="camera_frames_droid_manipverse_droid_replay")
parser.add_argument("--external-cam", default="cam__22008760")
parser.add_argument("--env-id", default="DROID-ManipVerse-Scene5")
parser.add_argument("--traj", default=None,
                    help="Path to droid_trajectory.pkl. Defaults to "
                         "<env.usd_file>.parent / droid_trajectory.pkl.")
parser.add_argument("--mp4-fps", type=float, default=15.0,
                    help="DROID's native control rate is 15 Hz.")
parser.add_argument("--substeps", type=int, default=1,
                    help="Replay each recorded step over N env.step calls.")
parser.add_argument("--start-from-recorded-q", action="store_true",
                    default=True,
                    help="Reset the env, then snap the Franka to "
                         "joint_positions[0] before stepping so the initial "
                         "control gap doesn't dominate the rollout.")
parser.add_argument("--no-pngs", dest="save_pngs", action="store_false")
parser.set_defaults(save_pngs=True)
parser.add_argument("--episode-length-s", type=float, default=180.0,
                    help="Override env_cfg.episode_length_s. Default 180s "
                         "comfortably covers a 60s DROID episode. The "
                         "polaris droid_cfg's own default is 30s which "
                         "truncates most DROID episodes mid-pick.")
parser.add_argument("--pd-params", default=None,
                    help="Path to tuned_pd_params.json (from stage 12). If "
                         "present, the env's actuator stiffness/damping is "
                         "overridden with these per-joint values before "
                         "rollout starts. Default: "
                         "<env.usd_file>.parent / tuned_pd_params.json if "
                         "it exists, otherwise leave env cfg untouched.")
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


PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


# --------------------------------------------------------------------------- #
# Image helpers (kept verbatim from manipverse_random_action.py for parity)
# --------------------------------------------------------------------------- #

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


def save_rgb(rgb, path: str):
    iio.imwrite(path, _to_uint8(rgb))


def combine_external_and_wrist(external, wrist):
    if external is None or wrist is None:
        return external if external is not None else wrist
    if external.shape[0] != wrist.shape[0]:
        from PIL import Image
        wrist_img = Image.fromarray(wrist)
        wrist = np.array(wrist_img.resize((wrist.shape[1], external.shape[0])))
    return np.concatenate((external, wrist), axis=1)


# --------------------------------------------------------------------------- #
# Trajectory loading
# --------------------------------------------------------------------------- #

def _resolve_traj_path(env_usd_file: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(env_usd_file).parent / "droid_trajectory.pkl"


def _load_replay(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"droid_trajectory.pkl not found at {path}\n"
            f"Run pipeline/11_droid_replay.py --scene <id> first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Robot state IO helpers (PolaRiS articulation has 7 arm joints + gripper)
# --------------------------------------------------------------------------- #

def _arm_indices(robot) -> list[int]:
    """Map the env's joint name list to panda_joint1..7 in order."""
    names = list(robot.data.joint_names)
    return [names.index(n) for n in PANDA_JOINT_NAMES]


def _read_arm_q(env: ManagerBasedRLSplatEnv) -> np.ndarray:
    robot = env.scene["robot"]
    idx = _arm_indices(robot)
    return robot.data.joint_pos[0, idx].detach().cpu().numpy()


def _resolve_pd_params_path(env_usd_file: str, override: str | None) -> Path | None:
    if override:
        p = Path(override).expanduser().resolve()
        return p if p.exists() else None
    p = Path(env_usd_file).parent / "tuned_pd_params.json"
    return p if p.exists() else None


def _apply_pd_overrides(env: "ManagerBasedRLSplatEnv", pd_path: Path) -> None:
    """Mutate the Franka articulation's actuator stiffness/damping tensors
    in-place from a stage-12 tuned_pd_params.json."""
    with open(pd_path) as f:
        pd = json.load(f)
    K = np.asarray(pd["stiffness"], dtype=np.float64)
    D = np.asarray(pd["damping"],   dtype=np.float64)
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    k_full = robot.data.default_joint_stiffness[0].detach().clone()
    d_full = robot.data.default_joint_damping[0].detach().clone()
    for i, name in enumerate(PANDA_JOINT_NAMES):
        j = joint_names.index(name)
        k_full[j] = float(K[i])
        d_full[j] = float(D[i])
    for actuator in robot.actuators.values():
        idx = actuator.joint_indices
        actuator.stiffness[:] = k_full[idx].unsqueeze(0).to(actuator.stiffness)
        actuator.damping[:]   = d_full[idx].unsqueeze(0).to(actuator.damping)
    print(f"[droid_replay] applied tuned PD gains from {pd_path}")
    print(f"               K = {K.tolist()}")
    print(f"               D = {D.tolist()}")


def _snap_arm_to(env: ManagerBasedRLSplatEnv, q: np.ndarray) -> None:
    """Force the articulation to a given joint state. Used to neutralise
    the start-of-episode control gap so the first few frames aren't a
    giant transient (the env always resets to the cfg's default pose,
    which is not the recorded q[0])."""
    robot = env.scene["robot"]
    idx = _arm_indices(robot)
    full_q = robot.data.joint_pos[0].detach().clone()
    full_q[idx] = torch.as_tensor(q, dtype=full_q.dtype, device=full_q.device)
    zero_v = torch.zeros_like(full_q)
    robot.write_joint_state_to_sim(
        full_q.unsqueeze(0), zero_v.unsqueeze(0),
        env_ids=torch.tensor([0], device=full_q.device),
    )


# --------------------------------------------------------------------------- #
# Action assembly (mirrors manipverse_rollout.py)
# --------------------------------------------------------------------------- #

def _action(joint_positions_t: np.ndarray, gripper_binary_t: float,
            device: str = "cuda") -> torch.Tensor:
    arm = torch.as_tensor(joint_positions_t, dtype=torch.float32, device=device)
    grip = torch.as_tensor([float(gripper_binary_t > 0.5)],
                           dtype=torch.float32, device=device)
    return torch.cat([arm, grip]).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.env_id, device="cuda:0",
                             num_envs=1, use_fabric=True)

    # The DROID-ManipVerse env defaults to episode_length_s=30, which at
    # 15 Hz caps every rollout at 450 steps. Real DROID episodes are
    # commonly 60–90 s. Stretch the budget so polaris's time_out
    # termination doesn't decapitate the replay halfway through.
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    print(f"[droid_replay] episode_length_s -> {env_cfg.episode_length_s:.1f}s "
          f"(default in droid_cfg is 30s)")

    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    try:
        language_instruction, initial_conditions = load_eval_initial_conditions(
            env.usd_file)
        print(f"[droid_replay] instruction: {language_instruction!r}")
        ic = initial_conditions[0] if initial_conditions else {}
    except Exception as e:
        print(f"[droid_replay] no eval ICs ({e}); using defaults")
        ic = {}

    print(f"[droid_replay] scene sensors: {list(env.scene.sensors.keys())}")

    traj_path = _resolve_traj_path(env.usd_file, args_cli.traj)
    print(f"[droid_replay] trajectory: {traj_path}")
    rep = _load_replay(traj_path)

    qs   = np.asarray(rep["joint_positions"], dtype=np.float32)
    gripb = np.asarray(rep.get("gripper_binary",
                       (rep["gripper_position"] > 0.5).astype(np.float32)),
                       dtype=np.float32)
    n_steps = qs.shape[0]
    print(f"[droid_replay] {n_steps} steps, dt≈{rep.get('dt', 1/15):.4f}s")

    env.reset(object_positions=ic)

    # PD overrides — picked up automatically from
    # <env.usd_file>.parent / tuned_pd_params.json if stage 12 has been
    # run + the file copied across (register_polaris_env.py --with-tuned-pd).
    pd_path = _resolve_pd_params_path(env.usd_file, args_cli.pd_params)
    if pd_path is not None:
        _apply_pd_overrides(env, pd_path)
    else:
        print(f"[droid_replay] no tuned_pd_params.json found — using default "
              f"PolaRiS PD gains (run stage 12 to tune).")

    if args_cli.start_from_recorded_q:
        _snap_arm_to(env, qs[0])
        print(f"[droid_replay] snapped Franka to recorded q[0]")

    external_key = args_cli.external_cam

    def grab_combined(obs_):
        splat = obs_.get("splat", {})
        ext = splat.get(external_key)
        wrist = splat.get("wrist_cam")
        return combine_external_and_wrist(_to_uint8(ext), _to_uint8(wrist))

    frames: list[np.ndarray] = []
    sim_qs: list[np.ndarray] = []
    obs = None
    frame_id = 0

    for t in range(n_steps):
        action = _action(qs[t], float(gripb[t]))
        for _ in range(max(1, int(args_cli.substeps))):
            obs, rew, term, trunc, info = env.step(action, expensive=True)

        sim_qs.append(_read_arm_q(env))
        combined = grab_combined(obs)
        if combined is not None:
            if args_cli.save_pngs:
                save_rgb(combined, str(save_dir / f"frame_{frame_id:06d}.png"))
            frames.append(combined)
            frame_id += 1

        if bool(term[0]) or bool(trunc[0]):
            print(f"[droid_replay] env terminated at step {t}/{n_steps}")
            break

    if frames:
        mp4_path = save_dir / "droid_replay.mp4"
        imageio_v2.mimsave(str(mp4_path), frames, fps=args_cli.mp4_fps)
        print(f"[droid_replay] {len(frames)} frames -> {mp4_path}")

    # Per-joint MSE between real and sim — the same metric stage 12
    # (sysid) minimises.
    sim_qs_arr = np.asarray(sim_qs, dtype=np.float64)
    real_qs_arr = qs[:sim_qs_arr.shape[0]].astype(np.float64)
    per_joint_mse = ((real_qs_arr - sim_qs_arr) ** 2).mean(axis=0).tolist()
    total_mse = float(((real_qs_arr - sim_qs_arr) ** 2).mean())
    print(f"[droid_replay] joint-tracking MSE: {total_mse:.6f} rad^2")

    rubric = info.get("rubric", {}) if obs is not None else {}
    summary = {
        "env_id":         args_cli.env_id,
        "trajectory":     str(traj_path),
        "n_steps":        int(n_steps),
        "n_frames":       int(len(frames)),
        "substeps":       int(args_cli.substeps),
        "per_joint_mse":  per_joint_mse,
        "total_mse":      total_mse,
        "rubric_success": rubric.get("success") if rubric else None,
        "rubric_progress": rubric.get("progress") if rubric else None,
    }
    (save_dir / "droid_replay_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"[droid_replay] summary -> {save_dir/'droid_replay_summary.json'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
