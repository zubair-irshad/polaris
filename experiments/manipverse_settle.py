"""REST3D-style physics settle for a ManipVerse PolaRiS scene.

Reconstructed objects are placed at their depth/ground-snap pose, which can
leave them penetrating the table plane, the background splat collider, or each
other (e.g. a marker resting inside a mug). When the scene is loaded with those
objects dynamic, PhysX resolves the overlap explosively and the object flies
off at episode start.

This worker borrows REST3D's "scene stabilization" idea
(github.com/ShirleyMaxx/REST3D — SAM 3D Objects + SAM 3 + Isaac Gym settle):
load the scene, drop the task objects under gravity against the *real*
collision geometry with depenetration capped, let them settle to a stable,
penetration-free resting configuration, then read the settled world poses back
out. Stage 07b (`pipeline/07b_settle.py`) bakes those poses into scene.usda +
initial_conditions.json so every eval starts from a stable rest state.

This is the manipverse-side IsaacSim harness; it has no manipverse import (same
contract as manipverse_random_action.py / manipverse_droid_replay.py). Run it
in the polaris env, e.g.:

    python experiments/manipverse_settle.py \
        --env-id DROID-ManipVerse-Scene1 \
        --out /abs/path/settled_poses.json \
        --settle-steps 240 --gpu 0
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

# --- IsaacSim app must launch before any IsaacLab-dependent import. ----------
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", default="DROID-ManipVerse-Scene1")
parser.add_argument("--out", required=True,
                    help="Where to write settled_poses.json.")
parser.add_argument("--settle-steps", type=int, default=240,
                    help="Physics steps to settle (default 240 ~= 1 s @ 240 Hz).")
parser.add_argument("--settle-tol", type=float, default=1e-4,
                    help="Early-stop when the max object linear speed (m/s) "
                         "drops below this for a few consecutive steps.")
parser.add_argument("--gpu", default=None,
                    help="Physical GPU index to pin the process to (sim + "
                         "torch). Defaults to CUDA_VISIBLE_DEVICES else '0'.")
args_cli, _ = parser.parse_known_args()

if args_cli.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"[settle] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
      flush=True)

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


def _load_base_ic(usd_file: str) -> dict:
    """Base (un-perturbed) object poses to settle from.

    Prefer the hub's initial_conditions.json (the single reconstructed "best"
    pose); fall back to trial 0 of the eval file; else {} (USD-authored poses).
    Pose format matches everything else in the pipeline: [x,y,z, qw,qx,qy,qz]
    (scalar-first quat, same as USD `quatd` and write_root_pose_to_sim).
    """
    base = Path(usd_file).parent / "initial_conditions.json"
    if base.exists():
        d = json.load(open(base))
        poses = d.get("poses") or []
        if poses:
            print(f"[settle] base poses <- {base}", flush=True)
            return poses[0]
    try:
        _, ics = load_eval_initial_conditions(usd_file)
        if ics:
            print("[settle] base poses <- eval initial conditions[0]", flush=True)
            return ics[0]
    except Exception as e:
        print(f"[settle] no eval ICs ({e})", flush=True)
    print("[settle] no IC file; settling from USD-authored poses", flush=True)
    return {}


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.env_id, device="cuda:0", num_envs=1,
                            use_fabric=True)
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore
    envu = env.unwrapped
    scene = envu.scene

    # Only pose objects that exist as scene entities. The base IC file can list
    # objects that aren't in the scene this run — e.g. a prim deactivated in
    # scene.usda (`active = false`) for an ablation, which the env never
    # registers. Posing an absent entity raises KeyError in reset().
    available = set(scene.rigid_objects)
    base_ic = _load_base_ic(envu.usd_file)
    dropped = [k for k in base_ic if k not in available]
    if dropped:
        print(f"[settle] dropping IC poses for absent scene entities: {dropped}",
              flush=True)
    base_ic = {k: v for k, v in base_ic.items() if k in available}

    # expensive=False: skip the splat render during settle (we only need physics).
    obs, info = env.reset(object_positions=base_ic, expensive=False)
    sim = envu.sim
    dt = sim.get_physics_dt()
    origin = scene.env_origins[0].detach().cpu().numpy()
    names = list(scene.rigid_objects)
    print(f"[settle] objects: {names}", flush=True)

    def world_pose(name):
        st = scene[name].data.root_state_w[0].detach().cpu().numpy()
        pos = st[:3] - origin                 # strip env origin -> base frame
        quat_wxyz = st[3:7]                    # IsaacLab root quat is wxyz
        return pos, quat_wxyz

    start = {n: world_pose(n)[0].copy() for n in names}

    # Hold the robot at its reset configuration so it can't disturb objects.
    robot_key = next((k for k in ("robot", "franka") if k in scene.articulations),
                     None)
    if robot_key is not None:
        rob = scene[robot_key]
        rob.set_joint_position_target(rob.data.joint_pos.clone())

    quiet = 0
    for i in range(args_cli.settle_steps):
        if robot_key is not None:
            scene[robot_key].set_joint_position_target(
                scene[robot_key].data.joint_pos.clone())
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        vmax = max(
            float(np.linalg.norm(
                scene[n].data.root_state_w[0, 7:10].detach().cpu().numpy()))
            for n in names) if names else 0.0
        quiet = quiet + 1 if vmax < args_cli.settle_tol else 0
        if i % 30 == 0 or quiet:
            print(f"[settle] step {i:4d}  max|v|={vmax:.5f} m/s", flush=True)
        if quiet >= 5:
            print(f"[settle] converged at step {i} (max|v|<{args_cli.settle_tol})",
                  flush=True)
            break

    settled = {}
    print("[settle] per-object displacement (settled - start):", flush=True)
    for n in names:
        pos, quat = world_pose(n)
        d = float(np.linalg.norm(pos - start[n]))
        print(f"[settle]   {n:28s} moved {d*1000:6.1f} mm  "
              f"-> z={pos[2]:.4f}", flush=True)
        settled[n] = [float(pos[0]), float(pos[1]), float(pos[2]),
                      float(quat[0]), float(quat[1]),
                      float(quat[2]), float(quat[3])]

    json.dump({"env_id": args_cli.env_id,
               "settle_steps": args_cli.settle_steps,
               "poses": settled}, open(out_path, "w"), indent=2)
    print(f"[settle] wrote {out_path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
