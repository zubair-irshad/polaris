import os

import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import numpy as np
import pandas as pd
from openpi_client import image_tools


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def main(eval_args: EvalArgs):
    # Robot rendering: the env reads POLARIS_ROBOT_SPLAT at gym.make time. Set
    # it explicitly (not just when unset) so the choice is deterministic even
    # if the parent shell exported it. "1" = gsplat (3DGS) robot (default),
    # "0" = synthetic IsaacSim-raytraced USD robot (--no-robot-splat).
    os.environ["POLARIS_ROBOT_SPLAT"] = "1" if eval_args.robot_splat else "0"
    print(
        f" >>> robot rendering: "
        f"{'gsplat (3DGS)' if eval_args.robot_splat else 'synthetic USD'} <<< "
    )

    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    # Optional episode-length override. max_episode_length (policy steps) is
    # derived as episode_length_s / (sim.dt * decimation), so raising this is
    # the only way to let a rollout run longer than the env default (30 s ->
    # 450 steps). Must be set on env_cfg BEFORE gym.make.
    if eval_args.episode_length_s is not None:
        env_cfg.episode_length_s = float(eval_args.episode_length_s)
        print(f" >>> episode_length_s override: {env_cfg.episode_length_s}s <<< ")
    env: MangerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    # CLI override: `--instruction "..."` wins over the instruction baked into
    # initial_conditions.json, so one env/USD can be evaluated on any prompt
    # without editing the JSON.
    if eval_args.instruction is not None:
        language_instruction = eval_args.instruction
    print(f" >>> language instruction: {language_instruction!r} <<< ")
    rollouts = len(initial_conditions)
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=initial_conditions[episode % len(initial_conditions)]
    )
    policy_client.reset()

    # Save EXACTLY what the policy ingests, for debugging. The DroidJointPos
    # client reads obs["splat"]["external_cam"] -> right/exterior_image and
    # obs["splat"]["wrist_cam"] -> wrist_image, each resize_with_pad'd to
    # 224x224 before being sent to the pi0 server (see droid_jointpos_client).
    # NOTE: obs has no "images" group and the cameras are named external_cam /
    # wrist_cam (not left/right/wrist_camera), so reading obs["images"] would
    # KeyError. We replicate the client's preprocessing so the saved feed is
    # pixel-for-pixel the policy input. Low-res (224) + temporally downsampled
    # via RGB_FEED_STRIDE to keep the file small.
    rgb_feed_path = run_folder / f"episode_{episode}_policy_input.mp4"
    rgb_feed = []

    def _policy_input_frame(obs):
        ext = image_tools.resize_with_pad(obs["splat"]["external_cam"], 224, 224)
        wrist = image_tools.resize_with_pad(obs["splat"]["wrist_cam"], 224, 224)
        return np.concatenate([ext, wrist], axis=1)  # [224, 448, 3], exterior | wrist

    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction)
        # `viz is not None` ONLY on chunk boundaries -- i.e. exactly the steps
        # where the client queried the server and `expensive`/splat rendering
        # was on, so obs["splat"] is the colored composite. On the in-between
        # open-loop steps `expensive=False`, obs["splat"] is the cheap gray
        # raw-mesh raster, and the policy never reads it. Gating the feed here
        # keeps it to the true (all-colored) policy inputs, one per chunk --
        # otherwise the mp4 flickers gray<->colored from the skipped renders.
        if viz is not None:
            video.append(viz)
            rgb_feed.append(_policy_input_frame(obs))

        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=eval_args.video_fps)

            # Save the exact policy-input feed for debugging.
            if rgb_feed:
                mediapy.write_video(rgb_feed_path, rgb_feed, fps=eval_args.video_fps)
                print(f" >>> saved policy-input feed: {rgb_feed_path} "
                      f"({len(rgb_feed)} frames, one per policy query) <<< ")

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)
        
            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(
                object_positions=initial_conditions[episode % len(initial_conditions)]
            )

            episode += 1
            video = []
            # Reset the policy-input feed for the next episode.
            rgb_feed_path = run_folder / f"episode_{episode}_policy_input.mp4"
            rgb_feed = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
