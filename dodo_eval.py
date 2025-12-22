### dodo_eval.py ###
import argparse
import os
import pickle
from importlib import metadata
import torch
import time  # <--- Added for real-time playback

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

from dodo_env_new_2 import Dodoenvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking-final")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("-v", "--vel", type=float, default=0.5)
    parser.add_argument("-y", "--yvel", type=float, default=0.0)
    parser.add_argument("-r", "--rot", type=float, default=0.0)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # We set min and max to the same value to force a constant command
    command_cfg["command_ranges"]["lin_vel_x"]   = [args.vel, args.vel]
    command_cfg["command_ranges"]["lin_vel_y"]   = [args.yvel, args.yvel]
    command_cfg["command_ranges"]["ang_vel_yaw"] = [args.rot, args.rot]
    
    # Increase resampling time so the command doesn't change during the demo
    command_cfg["resampling_time_s"] = 1000.0

    env = Dodoenvironment(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_name = f"model_{args.ckpt}.pt" if args.ckpt >= 0 else "model_final.pt"
    runner.load(os.path.join(log_dir, ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    print("\n=== Starting Evaluation ===")
    print(f"Commanding: Forward={args.vel} m/s, Side={args.yvel} m/s, Turn={args.rot} rad/s")

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            if env.episode_length_buf[0] % 100 == 0:
                print(f"Cmd: [{env.commands[0,0]:.2f}, {env.commands[0,1]:.2f}, {env.commands[0,2]:.2f}]")
                print(f"Vel: [{env.base_lin_vel[0,0]:.2f}, {env.base_lin_vel[0,1]:.2f}, {env.base_ang_vel[0,2]:.2f}]")

            # Slow down to real time (Simulation DT is usually 0.02s)
            #time.sleep(env.dt)

if __name__ == "__main__":
    main()
