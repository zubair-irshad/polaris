"""Random-action smoke test for the DROID-ManipVerse-Scene0 env.

Used to validate the gsplat 3DGS backend (POLARIS_RENDERER=gsplat, default)
against assets produced by the manipverse pipeline (Marble background +
SAM-3D-Objects rigid splats). Defaults to the synthetic USD robot
(POLARIS_ROBOT_SPLAT=0) since the splat robot asset is still 2DGS.

Usage (on the SSH GPU box, after `uv sync`):

    export MANIPVERSE_SCENE0_USD=/abs/path/to/scene_0/sim/scene.usda
    export POLARIS_ROBOT_SPLAT=0          # synthetic USD robot
    export POLARIS_RENDERER=gsplat        # default; explicit for clarity
    python experiments/manipverse_random_action.py \
        --save-dir camera_frames_droid_manipverse_scene0_gsplat
"""

import argparse
import os

import imageio.v3 as iio
import numpy as np
import torch
from isaaclab.app import AppLauncher

# --- IsaacSim app must be launched before any IsaacLab-dependent imports. ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-dir",
    default="camera_frames_droid_manipverse_scene0_gsplat",
    help="Where to write per-step PNGs.",
)
parser.add_argument(
    "--external-cam",
    default="cam__22008760",
    help="Name of the external camera in scene.sensors.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=200,
    help="Stop after this many steps even if the episode hasn't terminated.",
)
parser.add_argument(
    "--env-id",
    default="DROID-ManipVerse-Scene0",
)
args_cli, _ = parser.parse_known_args()
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
        wrist = np.array(
            wrist_img.resize((wrist.shape[1], external.shape[0]))
        )
    return np.concatenate((external, wrist), axis=1)


def main():
    save_dir = args_cli.save_dir
    os.makedirs(save_dir, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.env_id,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    try:
        language_instruction, initial_conditions = load_eval_initial_conditions(
            env.usd_file
        )
        print("Language instruction:", language_instruction)
        ic = initial_conditions[0]
    except Exception as e:
        print(f"[warn] no eval initial conditions found ({e}); resetting with defaults")
        ic = {}

    obs, info = env.reset(object_positions=ic)
    if isinstance(obs, dict):
        print("Top-level obs keys:", obs.keys())

    print("Available scene sensors:", list(env.scene.sensors.keys()))
    external_key = args_cli.external_cam

    def grab_combined(obs):
        splat = obs.get("splat", {})
        ext = splat.get(external_key)
        # Wrist key is registered under that exact name in droid_cfg.
        wrist = splat.get("wrist_cam")
        return combine_external_and_wrist(ext, wrist)

    frame_id = 0
    combined = grab_combined(obs)
    if combined is not None:
        save_rgb(combined, os.path.join(save_dir, f"frame_{frame_id:06d}.png"))
        frame_id += 1

    for _ in range(args_cli.max_steps):
        action = torch.tensor(env.action_space.sample(), device="cuda")
        obs, rew, term, trunc, info = env.step(action, expensive=True)

        combined = grab_combined(obs)
        if combined is not None:
            save_rgb(combined, os.path.join(save_dir, f"frame_{frame_id:06d}.png"))
            frame_id += 1

        if bool(term[0]) or bool(trunc[0]):
            break

    rubric = info.get("rubric", {})
    print(
        f"Episode finished. Success: {rubric.get('success')}, "
        f"Progress: {rubric.get('progress')}"
    )
    print(f"Saved {frame_id} RGB frames to: {save_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
