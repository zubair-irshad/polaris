"""Closed-loop pi0 / pi0.5 policy rollout on a DROID-ManipVerse-* env.

Sibling to ``manipverse_random_action.py`` (random policy), ``manipverse_rollout.py``
(cuRobo trajectory playback) and ``manipverse_orbit_capture.py`` (camera flythrough).
Same env loading, but the action stream comes from a learned OpenPI policy
(pi0_fast_droid_jointpos / pi05_droid_jointpos) served over a websocket — exactly
the eval path PolaRiS uses in ``scripts/eval.py``, just pointed at a manipverse
reconstructed scene instead of a hand-authored DROID-* task.

Architecture (unchanged from upstream PolaRiS — we only add the manipverse glue):

    serve_policy.py  ──websocket──▶  DroidJointPosClient  ──action(8,)──▶  env.step
        (OpenPI, separate venv)        (this script reuses it verbatim)

The policy consumes, per the DROID training format:
    observation/exterior_image_1_left  ← one external ZED view (224x224)
    observation/wrist_image_left        ← the gripper wrist camera (224x224)
    observation/joint_position          ← 7 panda joints
    observation/gripper_position        ← finger state in [0, 1]
    prompt                              ← language instruction
and returns 8-dim actions (7 absolute joint targets + 1 binary gripper),
chunked over ``--open-loop-horizon`` steps.

Camera glue
-----------
``DroidJointPosClient`` hardcodes ``obs["splat"]["external_cam"]`` and
``obs["splat"]["wrist_cam"]``. manipverse scenes register their two external
ZED cameras by serial (``cam_22008760``, ``cam_24400334``) plus ``wrist_cam``.
So before each ``infer`` we alias the chosen serial → ``external_cam`` (the
client is left untouched). ``--external-cam`` selects which of the two ZED views
the policy treats as ``exterior_image_1_left``; pi0 was trained on a single,
randomly-chosen exterior view per episode, so either works — we default to the
left camera and expose the flag for A/B comparison.

Server (launch separately, on the box's GPU, before this script):

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
      external/polaris/third_party/openpi/.venv/bin/python \
      external/polaris/third_party/openpi/scripts/serve_policy.py \
      --port 8000 \
      policy:checkpoint --policy.config pi05_droid_jointpos_polaris \
      --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris

Then, in the polaris (`uv sync`-ed) venv:

    export POLARIS_ROBOT_SPLAT=0
    export POLARIS_RENDERER=gsplat
    python experiments/manipverse_pi0_rollout.py \
        --env-id DROID-ManipVerse-Scene5 \
        --external-cam cam_22008760 \
        --host 0.0.0.0 --port 8000 \
        --save-dir camera_frames_scene5_pi0

Outputs (under --save-dir):
    scene_view/frame_XXXXXX.png   full-res external+wrist composite (splat render)
    scene_view.mp4                MP4 of the above
    model_view/frame_XXXXXX.png   exactly what pi0 sees: 224x224 exterior | 224x224 wrist
    model_view.mp4                MP4 of the above
    pi0_rollout_summary.json      instruction, cams, steps, success, progress
"""

import argparse
import json
import os
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
    default="camera_frames_droid_manipverse_pi0",
    help="Where to write per-step PNGs + MP4s + summary JSON.",
)
parser.add_argument(
    "--env-id",
    default="DROID-ManipVerse-Scene5",
)
parser.add_argument(
    "--external-cam",
    default="cam_22008760",
    help="Scene camera serial fed to the policy as exterior_image_1_left. "
         "manipverse scenes expose two: cam_22008760 (left) and cam_24400334 "
         "(right). Aliased to the client's hardcoded 'external_cam' key.",
)
parser.add_argument(
    "--wrist-cam",
    default="wrist_cam",
    help="Scene camera key fed to the policy as wrist_image_left.",
)
# --- policy server / client knobs (mirror polaris.config.PolicyArgs) ---
parser.add_argument("--client", default="DroidJointPos",
                    help="Registered InferenceClient name (DroidJointPos, Fake).")
parser.add_argument("--host", default="0.0.0.0", help="OpenPI policy server host.")
parser.add_argument("--port", type=int, default=8000, help="OpenPI policy server port.")
parser.add_argument("--open-loop-horizon", type=int, default=8,
                    help="Action-chunk horizon: execute N predicted actions "
                         "before re-querying the server (pi0 default 8).")
parser.add_argument("--instruction", default=None,
                    help="Override the language instruction. Defaults to the "
                         "scene's initial_conditions instruction.")
parser.add_argument("--rollouts", type=int, default=1,
                    help="Number of eval episodes, each reset to a different "
                         "initial condition from the IC file's 'poses' list "
                         "(PolaRiS-style multi-trial eval). Capped at the "
                         "number of available poses. Default 1.")
parser.add_argument("--initial-conditions-file", default=None,
                    help="Path to an initial_conditions.json with a 'poses' "
                         "list of >=1 initial conditions (e.g. the perturbed "
                         "initial_conditions_eval.json from "
                         "scripts/make_eval_initial_conditions.py). Default: "
                         "initial_conditions.json next to the env USD.")
parser.add_argument("--max-steps", type=int, default=400,
                    help="Hard step cap (also bounded by env.max_episode_length).")
parser.add_argument("--mp4-fps", type=float, default=15.0,
                    help="Output MP4 frame rate (eval.py uses 15).")
parser.add_argument(
    "--faithful-render", action="store_true",
    help="Respect the policy's rerender flag (cheap sim-only render on "
         "open-loop steps) — matches the optimized eval path but produces "
         "non-composited frames on those steps. Default forces a full splat "
         "composite every step for clean visualization.",
)
parser.add_argument(
    "--no-pngs", dest="save_pngs", action="store_false",
    help="Skip per-step PNG dumps; write only the MP4s.",
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
    f"[pi0] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
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
from polaris.config import PolicyArgs  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import (  # noqa: E402
    ManagerBasedRLSplatEnv,
)
from polaris.policy import InferenceClient  # noqa: E402
from polaris.utils import load_eval_initial_conditions  # noqa: E402


# --------------------------------------------------------------------------- #
# Image helpers (copied from manipverse_rollout.py for parity)
# --------------------------------------------------------------------------- #

def save_rgb(rgb, path: str):
    rgb = _to_uint8(rgb)
    if rgb is not None:
        iio.imwrite(path, rgb)


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


def combine_external_and_wrist(external, wrist):
    if external is None or wrist is None:
        return external if external is not None else wrist
    if external.shape[0] != wrist.shape[0]:
        from PIL import Image
        wrist_img = Image.fromarray(wrist)
        wrist = np.array(wrist_img.resize((wrist.shape[1], external.shape[0])))
    return np.concatenate((external, wrist), axis=1)


# --------------------------------------------------------------------------- #
# Observation glue: alias the chosen ZED serial -> the key the client expects.
# DroidJointPosClient._extract_observation reads obs["splat"]["external_cam"]
# and obs["splat"]["wrist_cam"]; manipverse scenes key cameras by serial.
# --------------------------------------------------------------------------- #

def _alias_obs_for_client(obs: dict, external_serial: str, wrist_key: str) -> dict:
    splat = obs.get("splat")
    if not isinstance(splat, dict):
        raise KeyError(
            "obs has no 'splat' dict — did you call env.step/reset with "
            "expensive rendering enabled?"
        )
    if external_serial not in splat:
        raise KeyError(
            f"external camera {external_serial!r} not in rendered cams "
            f"{list(splat.keys())}. Pass --external-cam <serial> matching one "
            f"of the scene's UsdGeom.Camera prim names."
        )
    if wrist_key not in splat:
        raise KeyError(
            f"wrist camera {wrist_key!r} not in rendered cams "
            f"{list(splat.keys())}."
        )
    # Client keys, set explicitly (overwrite any fallback external_cam).
    splat["external_cam"] = splat[external_serial]
    if wrist_key != "wrist_cam":
        splat["wrist_cam"] = splat[wrist_key]
    return obs


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def _run_episode(env, policy_client, language_instruction, ic, horizon,
                 save_dir, episode, save_pngs, single):
    """Run one closed-loop rollout from initial condition `ic`.

    Returns (steps, success, progress). Writes per-episode MP4s (and, when
    `save_pngs`, per-step PNGs into an episode subdir). When `single` (the
    legacy 1-rollout path) the MP4s keep their historical scene_view.mp4 /
    model_view.mp4 names and PNGs land in flat scene_view/ + model_view/ dirs.
    """
    def grab_scene(obs_):
        splat = obs_.get("splat", {})
        ext = splat.get(args_cli.external_cam)
        wrist = splat.get(args_cli.wrist_cam)
        return combine_external_and_wrist(_to_uint8(ext), _to_uint8(wrist))

    if single:
        scene_mp4 = save_dir / "scene_view.mp4"
        model_mp4 = save_dir / "model_view.mp4"
        scene_png_dir = save_dir / "scene_view"
        model_png_dir = save_dir / "model_view"
    else:
        scene_mp4 = save_dir / f"episode_{episode}_scene.mp4"
        model_mp4 = save_dir / f"episode_{episode}_model.mp4"
        scene_png_dir = save_dir / f"episode_{episode}" / "scene_view"
        model_png_dir = save_dir / f"episode_{episode}" / "model_view"
    if save_pngs:
        scene_png_dir.mkdir(parents=True, exist_ok=True)
        model_png_dir.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset(object_positions=ic)
    policy_client.reset()

    scene_frames: list[np.ndarray] = []
    model_frames: list[np.ndarray] = []
    step = 0
    last_info = info
    while True:
        _alias_obs_for_client(obs, args_cli.external_cam, args_cli.wrist_cam)
        # return_viz=True => `viz` is the 224x(2*224) the policy actually sees.
        action, viz = policy_client.infer(obs, language_instruction, return_viz=True)

        if viz is not None:
            viz = _to_uint8(viz)
            model_frames.append(viz)
            if save_pngs:
                save_rgb(viz, str(model_png_dir / f"frame_{step:06d}.png"))

        scene = grab_scene(obs)
        if scene is not None:
            scene_frames.append(scene)
            if save_pngs:
                save_rgb(scene, str(scene_png_dir / f"frame_{step:06d}.png"))

        # Default: force a full splat composite every step for clean viz.
        # --faithful-render respects the chunked-policy rerender optimization.
        expensive = policy_client.rerender if args_cli.faithful_render else True
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action, dtype=torch.float32,
                         device=env.unwrapped.device).reshape(1, -1),
            expensive=expensive,
        )
        last_info = info
        step += 1

        if bool(term[0]) or bool(trunc[0]) or step >= horizon:
            break

    # Capture the final observation frame too.
    _alias_obs_for_client(obs, args_cli.external_cam, args_cli.wrist_cam)
    scene = grab_scene(obs)
    if scene is not None:
        scene_frames.append(scene)
        if save_pngs:
            save_rgb(scene, str(scene_png_dir / f"frame_{step:06d}.png"))

    if scene_frames:
        imageio_v2.mimsave(str(scene_mp4), scene_frames, fps=args_cli.mp4_fps)
    if model_frames:
        imageio_v2.mimsave(str(model_mp4), model_frames, fps=args_cli.mp4_fps)

    rubric = last_info.get("rubric", {}) if isinstance(last_info, dict) else {}
    success = rubric.get("success")
    progress = rubric.get("progress")
    print(f"[pi0] episode {episode}: steps={step} success={success} "
          f"progress={progress} -> {scene_mp4.name}")
    return int(step), success, progress


def main():
    import csv

    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.env_id,
        device="cuda:0",  # concrete index; process is pinned to one GPU above
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    # Instruction + initial conditions (mirrors manipverse_random_action.py).
    # With --initial-conditions-file + --rollouts this loads the PolaRiS-style
    # multi-trial 'poses' list (e.g. the perturbed initial_conditions_eval.json).
    try:
        language_instruction, initial_conditions = load_eval_initial_conditions(
            usd=env.usd_file,
            initial_conditions_file=args_cli.initial_conditions_file,
            rollouts=args_cli.rollouts,
        )
        if not initial_conditions:
            initial_conditions = [{}]
    except Exception as e:
        print(f"[pi0] no eval ICs ({e}); using defaults")
        language_instruction, initial_conditions = "", [{}]
    if args_cli.instruction:
        language_instruction = args_cli.instruction

    rollouts = min(int(args_cli.rollouts), len(initial_conditions))
    single = rollouts <= 1
    # PNGs per step balloon over 100+ rollouts; only honor them for a single
    # rollout (multi-trial eval keeps just the per-episode MP4s + CSV).
    save_pngs = args_cli.save_pngs and single

    print(f"[pi0] instruction: {language_instruction!r}")
    print(f"[pi0] scene sensors: {list(env.scene.sensors.keys())}")
    print(f"[pi0] external cam -> policy exterior: {args_cli.external_cam}")
    print(f"[pi0] wrist cam     -> policy wrist:    {args_cli.wrist_cam}")
    print(f"[pi0] rollouts: {rollouts} (poses available: {len(initial_conditions)})")
    if args_cli.save_pngs and not save_pngs:
        print("[pi0] (per-step PNGs disabled for multi-rollout eval; MP4s only)")

    # Policy client — same registry/path as scripts/eval.py.
    policy_args = PolicyArgs(
        client=args_cli.client,
        host=args_cli.host,
        port=args_cli.port,
        open_loop_horizon=args_cli.open_loop_horizon,
    )
    print(f"[pi0] connecting client={args_cli.client} -> "
          f"{args_cli.host}:{args_cli.port} (horizon={args_cli.open_loop_horizon})")
    policy_client: InferenceClient = InferenceClient.get_client(policy_args)

    horizon = min(int(args_cli.max_steps), int(env.max_episode_length))

    # Resumable per-episode CSV (mirrors scripts/eval.py): skip episodes that
    # already have a row, so a crashed 100-rollout run picks up where it left.
    csv_path = save_dir / "eval_results.csv"
    fieldnames = ["episode", "episode_length", "success", "progress"]
    done = 0
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        done = len(rows)
        if done:
            print(f"[pi0] resuming: {done} episode(s) already in {csv_path.name}")

    for episode in range(done, rollouts):
        ic = initial_conditions[episode % len(initial_conditions)]
        steps, success, progress = _run_episode(
            env, policy_client, language_instruction, ic, horizon,
            save_dir, episode, save_pngs, single)
        rows.append({
            "episode": episode,
            "episode_length": steps,
            "success": success,
            "progress": progress,
        })
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def _truthy(v):
        return str(v).strip().lower() in ("true", "1", "1.0")

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    successes = [r for r in rows if _truthy(r["success"])]
    progresses = [p for p in (_num(r["progress"]) for r in rows) if p is not None]
    n = len(rows)
    summary = {
        "env_id":            args_cli.env_id,
        "instruction":       language_instruction,
        "client":            args_cli.client,
        "host":              args_cli.host,
        "port":              args_cli.port,
        "open_loop_horizon": args_cli.open_loop_horizon,
        "external_cam":      args_cli.external_cam,
        "wrist_cam":         args_cli.wrist_cam,
        "initial_conditions_file": args_cli.initial_conditions_file,
        "rollouts":          n,
        "num_success":       len(successes),
        "success_rate":      (len(successes) / n) if n else None,
        "mean_progress":     (sum(progresses) / len(progresses)) if progresses else None,
        "episodes":          rows,
    }
    (save_dir / "pi0_eval_summary.json").write_text(json.dumps(summary, indent=2))
    if single:
        # Preserve the historical single-rollout summary name too.
        (save_dir / "pi0_rollout_summary.json").write_text(
            json.dumps(summary, indent=2))

    print(f"\n[pi0] ===== eval over {n} rollout(s) =====")
    print(f"[pi0] success rate: {len(successes)}/{n}"
          + (f" ({100*len(successes)/n:.1f}%)" if n else ""))
    if progresses:
        print(f"[pi0] mean progress: {sum(progresses)/len(progresses):.3f}")
    print(f"[pi0] per-episode CSV -> {csv_path}")
    print(f"[pi0] summary -> {save_dir/'pi0_eval_summary.json'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
