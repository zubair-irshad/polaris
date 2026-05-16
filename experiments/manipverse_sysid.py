"""System-identification driver — tunes the IsaacSim Franka's per-joint
PD gains to match the recorded DROID joint trajectory.

This is the IsaacLab side of manipverse's stage 12. The manipverse
wrapper (``pipeline/12_sysid.py``) ships ``droid_trajectory.pkl`` to
``<hub>/<env>/`` and invokes us with ``--env-id``, the initial PD
vector, CMA-ES options, and a target JSON path. We:

  1. Launch IsaacSim once (very expensive — single startup).
  2. Build the DROID-ManipVerse env, parameterised at construction time
     by a closure that overrides the Franka ImplicitActuatorCfg with
     per-joint stiffness/damping dicts.
  3. Drive ``manipverse.sim.sysid.system_identification`` (CMA-ES) and
     score each sample by simulating the replay (joint targets =
     recorded q[t]) and computing MSE against the recorded q.
  4. Write the best vector to ``--out-json``.

To swap the Franka's PD gains between evaluations we have two options:

  A. Rebuild the env (closes + re-makes the Articulation). Robust but
     slow (~seconds per eval).
  B. Patch ``robot.actuators[*].stiffness/damping`` tensors in-place
     after env construction. ~10x faster, no kit reload. Risk is small
     because ImplicitActuator's gains are mutable torch tensors that
     PhysX reads at every step.

We use (B). If you observe weirdness, pass ``--rebuild-each`` to fall
back to (A).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", default="DROID-ManipVerse-Scene5")
parser.add_argument("--traj", default=None,
                    help="Path to droid_trajectory.pkl. Defaults to "
                         "<env.usd_file>.parent / droid_trajectory.pkl.")
parser.add_argument("--out-json", required=True,
                    help="Where to write tuned_pd_params.json.")
parser.add_argument("--x0-stiffness", required=True,
                    help="Comma-separated 7 stiffness values (panda_joint1..7).")
parser.add_argument("--x0-damping", required=True,
                    help="Comma-separated 7 damping values (panda_joint1..7).")
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--maxiter", type=int, default=30)
parser.add_argument("--popsize", type=int, default=0)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--rollout-steps", type=int, default=0)
parser.add_argument("--rebuild-each", action="store_true", default=False,
                    help="Rebuild env per eval instead of patching gains "
                         "in-place. Slower but more bullet-proof.")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False  # sysid doesn't need rendering
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# --------------------------------------------------------------------------- #

import gymnasium as gym  # noqa: E402

import polaris.environments  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import (  # noqa: E402
    ManagerBasedRLSplatEnv,
)
from polaris.utils import load_eval_initial_conditions  # noqa: E402

# manipverse helpers — manipverse_sysid is launched with PYTHONPATH
# containing the manipverse repo root (see pipeline/12_sysid.py).
from manipverse.sim.sysid import (  # noqa: E402
    PDParams, system_identification, write_tuned_params,
)
from manipverse.sim.droid_replay import joint_tracking_loss  # noqa: E402


PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _parse_csv(s: str, n: int) -> np.ndarray:
    arr = np.asarray([float(x) for x in s.split(",") if x.strip()],
                     dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"expected {n} comma-separated values, got {arr.size}")
    return arr


def _resolve_traj_path(env_usd_file: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(env_usd_file).parent / "droid_trajectory.pkl"


def _load_replay(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"droid_trajectory.pkl not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Robot state IO
# --------------------------------------------------------------------------- #

def _arm_indices(robot) -> list[int]:
    names = list(robot.data.joint_names)
    return [names.index(n) for n in PANDA_JOINT_NAMES]


def _read_arm_q(env) -> np.ndarray:
    robot = env.scene["robot"]
    idx = _arm_indices(robot)
    return robot.data.joint_pos[0, idx].detach().cpu().numpy()


def _snap_arm_to(env, q: np.ndarray) -> None:
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
# PD-gain mutation
# --------------------------------------------------------------------------- #

def _apply_pd(env, params: PDParams) -> None:
    """In-place override of stiffness/damping tensors for every actuator
    on the robot Articulation. ImplicitActuator stores these as
    ``self.stiffness`` / ``self.damping`` torch tensors of shape
    (num_envs, num_joints_in_group); PhysX reads them every step.
    """
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    # Build a per-joint lookup, falling back to the actuator's current
    # value for non-arm joints (e.g. finger_joint).
    k_full = robot.data.default_joint_stiffness[0].detach().clone()
    d_full = robot.data.default_joint_damping[0].detach().clone()
    for i, name in enumerate(PANDA_JOINT_NAMES):
        j = joint_names.index(name)
        k_full[j] = float(params.stiffness[i])
        d_full[j] = float(params.damping[i])

    # Push to every actuator group: each group holds a (num_envs, k) slice
    # of the full joint vector. We index into k_full by the group's
    # joint_indices.
    for actuator in robot.actuators.values():
        idx = actuator.joint_indices
        actuator.stiffness[:] = k_full[idx].unsqueeze(0).to(actuator.stiffness)
        actuator.damping[:]   = d_full[idx].unsqueeze(0).to(actuator.damping)


def _action(joint_positions_t: np.ndarray, gripper_binary_t: float,
            device: str = "cuda") -> torch.Tensor:
    arm = torch.as_tensor(joint_positions_t, dtype=torch.float32, device=device)
    grip = torch.as_tensor([float(gripper_binary_t > 0.5)],
                           dtype=torch.float32, device=device)
    return torch.cat([arm, grip]).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #

def _rollout_and_score(env, params: PDParams, real_q: np.ndarray,
                        grip_bin: np.ndarray, ic: dict, warmup: int,
                        n_steps: int) -> tuple[float, np.ndarray]:
    env.reset(object_positions=ic)
    _apply_pd(env, params)
    _snap_arm_to(env, real_q[0])

    sim_qs = np.zeros_like(real_q[:n_steps])
    for t in range(n_steps):
        action = _action(real_q[t], float(grip_bin[t]))
        env.step(action, expensive=False)
        sim_qs[t] = _read_arm_q(env)

    return joint_tracking_loss(real_q[:n_steps], sim_qs, warmup=warmup), sim_qs


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    K0 = _parse_csv(args_cli.x0_stiffness, 7)
    D0 = _parse_csv(args_cli.x0_damping,   7)
    x0 = np.concatenate([K0, D0])

    env_cfg = parse_env_cfg(args_cli.env_id, device="cuda",
                             num_envs=1, use_fabric=True)
    env: ManagerBasedRLSplatEnv = gym.make(args_cli.env_id, cfg=env_cfg)  # type: ignore

    try:
        _, initial_conditions = load_eval_initial_conditions(env.usd_file)
        ic = initial_conditions[0] if initial_conditions else {}
    except Exception as e:
        print(f"[sysid] no eval ICs ({e}); using defaults")
        ic = {}

    traj_path = _resolve_traj_path(env.usd_file, args_cli.traj)
    print(f"[sysid] trajectory: {traj_path}")
    rep = _load_replay(traj_path)

    real_q = np.asarray(rep["joint_positions"], dtype=np.float32)
    grip_b = np.asarray(rep.get("gripper_binary",
                                (rep["gripper_position"] > 0.5).astype(np.float32)),
                        dtype=np.float32)
    n_total = real_q.shape[0]
    n_steps = (int(args_cli.rollout_steps)
               if args_cli.rollout_steps > 0
               else n_total)
    n_steps = min(n_steps, n_total)
    print(f"[sysid] using {n_steps}/{n_total} recorded steps for the loss "
          f"(warmup={args_cli.warmup})")

    eval_counter = {"n": 0}
    def replay_fn(params: PDParams) -> float:
        eval_counter["n"] += 1
        loss, _ = _rollout_and_score(
            env, params, real_q, grip_b, ic,
            warmup=args_cli.warmup, n_steps=n_steps,
        )
        if eval_counter["n"] % 5 == 0:
            print(f"[sysid]   eval #{eval_counter['n']}: loss={loss:.6f}  "
                  f"K={np.round(params.stiffness, 1).tolist()}  "
                  f"D={np.round(params.damping, 1).tolist()}")
        return loss

    print(f"[sysid] starting CMA-ES  sigma={args_cli.sigma}  maxiter={args_cli.maxiter}")
    result = system_identification(
        replay_fn, x0=x0,
        sigma=args_cli.sigma,
        maxiter=args_cli.maxiter,
        popsize=(args_cli.popsize or None),
    )

    # Re-score the best params with the warmup-trimmed loss + record the
    # baseline (x0) loss for context. Useful to know if sysid actually
    # improved things.
    best = PDParams.from_vector(np.concatenate([result["stiffness"],
                                                 result["damping"]]))
    best_loss, _ = _rollout_and_score(env, best, real_q, grip_b, ic,
                                       warmup=args_cli.warmup, n_steps=n_steps)
    baseline = PDParams.from_vector(x0)
    baseline_loss, _ = _rollout_and_score(env, baseline, real_q, grip_b, ic,
                                           warmup=args_cli.warmup,
                                           n_steps=n_steps)
    print(f"[sysid] baseline loss = {baseline_loss:.6f}  "
          f"best loss = {best_loss:.6f}  "
          f"improvement = {(baseline_loss - best_loss)/max(baseline_loss, 1e-12)*100:.1f}%")

    tuned = best.to_dict()
    tuned.update({
        "best_loss":     float(best_loss),
        "baseline_loss": float(baseline_loss),
        "iterations":    result["iterations"],
        "rollout_steps": int(n_steps),
        "warmup":        int(args_cli.warmup),
        "env_id":        args_cli.env_id,
        "trajectory":    str(traj_path),
    })
    write_tuned_params(Path(args_cli.out_json), tuned)
    print(f"[sysid] wrote {args_cli.out_json}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
