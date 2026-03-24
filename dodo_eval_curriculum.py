import argparse
import os
import pickle
from importlib import metadata
import torch
import time
import numpy as np

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# MAKE SURE THIS MATCHES YOUR FILENAME
from dodo_env_new_2 import Dodoenvironment 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking-final")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("-v", "--vel", type=float, default=0.5)
    parser.add_argument("-y", "--yvel", type=float, default=0.0)
    parser.add_argument("-r", "--rot", type=float, default=0.0)
    # NEW: Allow choosing the terrain difficulty (0-9)
    parser.add_argument("-l", "--level", type=int, default=0, help="Terrain difficulty level (0-9)")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init()

    log_dir = f"logs/{args.exp_name}"
    # Load the exact configs used during training
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # Force constant commands for the demo
    command_cfg["command_ranges"]["lin_vel_x"]   = [args.vel, args.vel]
    command_cfg["command_ranges"]["lin_vel_y"]   = [args.yvel, args.yvel]
    command_cfg["command_ranges"]["ang_vel_yaw"] = [args.rot, args.rot]
    command_cfg["resampling_time_s"] = 1000.0 # Don't change commands

    # Create the environment with Viewer enabled
    env = Dodoenvironment(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load the trained agent
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_name = f"model_{args.ckpt}.pt" if args.ckpt >= 0 else "model_final.pt"
    runner.load(os.path.join(log_dir, ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)

    # --- CRITICAL FIX FOR TERRAIN VIEWING ---
    # 1. Force the terrain level to what you requested
    env.terrain_levels[:] = args.level
    
    # 2. Reset the environment (this will spawn the robot at the correct X location for that level)
    obs, _ = env.reset()

    print("\n=== Starting Evaluation ===")
    print(f"Terrain Level: {args.level}")
    print(f"Commanding: Forward={args.vel} m/s, Side={args.yvel} m/s, Turn={args.rot} rad/s")

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            
            # If the robot finishes the strip or falls, force it back to the requested level
            if dones.any():
                env.terrain_levels[:] = args.level
                
            if env.episode_length_buf[0] % 50 == 0:
                print(f"Cmd: [{env.commands[0,0]:.2f}, {env.commands[0,1]:.2f}, {env.commands[0,2]:.2f}] "
                      f"| Vel: [{env.base_lin_vel[0,0]:.2f}] "
                      f"| Height: {env.base_pos[0,2]:.2f}")

            # Optional: Add sleep to watch it in real-time (simulation is faster than real life)
            # time.sleep(0.02)

if __name__ == "__main__":
    main()