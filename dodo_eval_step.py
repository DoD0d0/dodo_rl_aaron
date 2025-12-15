### dodo_eval_step.py ###
import argparse
import os
import pickle
from importlib import metadata
import torch
import numpy as np
import math

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
from dodo_env import DodoEnv


def print_separator(title):
    print("=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_key_parameters(env, step):
    """Print all key parameters used in reward calculations"""
    print(f"\n--- STEP {step} KEY PARAMETERS ---")
    
    # Commands (targets)
    print(f"Commands (targets):")
    print(f"  lin_vel_x_target: {env.commands[0,0]:.4f} m/s")
    print(f"  lin_vel_y_target: {env.commands[0,1]:.4f} m/s") 
    print(f"  ang_vel_z_target: {env.commands[0,2]:.4f} rad/s")
    
    # Base state
    print(f"Base State:")
    print(f"  base_pos: [{env.base_pos[0,0]:.4f}, {env.base_pos[0,1]:.4f}, {env.base_pos[0,2]:.4f}] m")
    print(f"  base_lin_vel: [{env.base_lin_vel[0,0]:.4f}, {env.base_lin_vel[0,1]:.4f}, {env.base_lin_vel[0,2]:.4f}] m/s")
    print(f"  base_ang_vel: [{env.base_ang_vel[0,0]:.4f}, {env.base_ang_vel[0,1]:.4f}, {env.base_ang_vel[0,2]:.4f}] rad/s")
    print(f"  base_euler (deg): [{env.base_euler[0,0]:.2f}, {env.base_euler[0,1]:.2f}, {env.base_euler[0,2]:.2f}]")
    
    # Joint states
    print(f"Joint States:")
    joint_names = env.env_cfg["joint_names"]
    for i, name in enumerate(joint_names):
        print(f"  {name:15s}: pos={env.dof_pos[0,i]:+7.4f} rad, vel={env.dof_vel[0,i]:+7.4f} rad/s, default={env.default_dof_pos[i]:+7.4f}")
    
    # Actions
    print(f"Actions:")
    print(f"  current_actions:  {[f'{a:.4f}' for a in env.actions[0].cpu().numpy()]}")
    print(f"  last_actions:     {[f'{a:.4f}' for a in env.last_actions[0].cpu().numpy()]}")
    
    # Special joint indices values
    print(f"Special Joint Values:")
    if len(env.hip_aa_indices) > 0:
        hip_aa_vals = [env.dof_pos[0, idx].item() for idx in env.hip_aa_indices]
        print(f"  hip_aa_values: {[f'{v:.4f}' for v in hip_aa_vals]}")
    
    if len(env.hip_fe_indices) > 0:
        hip_fe_vals = [env.dof_pos[0, idx].item() for idx in env.hip_fe_indices]
        print(f"  hip_fe_values: {[f'{v:.4f}' for v in hip_fe_vals]}")
        print(f"  hip_fe_diff: {abs(hip_fe_vals[0] - hip_fe_vals[1]):.4f}")
    
    if len(env.knee_fe_indices) > 0:
        knee_fe_vals = [env.dof_pos[0, idx].item() for idx in env.knee_fe_indices]
        print(f"  knee_fe_values: {[f'{v:.4f}' for v in knee_fe_vals]}")
    
    # Ankle heights and foot orientations
    if hasattr(env, 'current_ankle_heights'):
        print(f"  ankle_heights: {[f'{h:.4f}' for h in env.current_ankle_heights[0].cpu().numpy()]} m")
        print(f"  mean_ankle_height: {env.current_ankle_heights[0].mean().item():.4f} m")
    
    if hasattr(env, 'current_foot_orientations'):
        print(f"Foot Orientations (quaternions):")
        for i, foot_name in enumerate(['right_foot', 'left_foot']):
            quat = env.current_foot_orientations[0, i].cpu().numpy()
            print(f"  {foot_name}: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")


def print_reward_values(env, step):
    """Calculate and print all reward function values"""
    print(f"\n--- STEP {step} REWARD VALUES ---")
    
    total_reward = 0.0
    
    # Calculate each reward component
    for reward_name, scale in env.reward_scales.items():
        if hasattr(env, f'_reward_{reward_name}'):
            reward_fn = getattr(env, f'_reward_{reward_name}')
            raw_reward = reward_fn()
            scaled_reward = raw_reward * scale
            
            if torch.is_tensor(raw_reward):
                raw_val = raw_reward[0].item() if raw_reward.dim() > 0 else raw_reward.item()
                scaled_val = scaled_reward[0].item() if scaled_reward.dim() > 0 else scaled_reward.item()
            else:
                raw_val = raw_reward
                scaled_val = scaled_reward
            
            total_reward += scaled_val
            
            print(f"  {reward_name:25s}: raw={raw_val:+8.4f}, scale={scale:+8.4f}, scaled={scaled_val:+8.4f}")
    
    print(f"  {'TOTAL_REWARD':25s}: {total_reward:+8.4f}")
    
    # Print reward calculation details for some key rewards
    print(f"\nReward Calculation Details:")
    
    # Tracking rewards details
    lin_vel_err = torch.sum((env.commands[0,:2] - env.base_lin_vel[0,:2])**2).item()
    ang_vel_err = (env.commands[0,2] - env.base_ang_vel[0,2])**2
    print(f"  lin_vel_tracking_error: {lin_vel_err:.6f}")
    print(f"  ang_vel_tracking_error: {ang_vel_err:.6f}")
    print(f"  tracking_sigma: {env.reward_cfg.get('tracking_sigma', 'N/A')}")
    
    # Base height details
    height_target = env.reward_cfg.get('base_height_target', 0.48)
    height_error = (env.base_pos[0,2] - height_target)**2
    print(f"  base_height_error: {height_error:.6f} (target: {height_target:.3f}m, actual: {env.base_pos[0,2]:.3f}m)")
    
    # Orientation stability details
    pitch_rad = abs(env.base_euler[0,1] * math.pi/180)
    roll_rad = abs(env.base_euler[0,0] * math.pi/180)
    print(f"  orientation_error: pitch²+roll² = {pitch_rad**2:.6f} + {roll_rad**2:.6f} = {pitch_rad**2 + roll_rad**2:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking")
    parser.add_argument("--ckpt", type=int, default=-1)  # -1 for final model
    parser.add_argument("-v", "--vel", type=float, default=0.5)
    parser.add_argument("-y", "--yvel", type=float, default=0.0)
    parser.add_argument("-r", "--rot", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # Fallback für alte Key-Namen
    if "resampling_time_s" not in command_cfg and "resampling_time" in command_cfg:
        command_cfg["resampling_time_s"] = command_cfg.pop("resampling_time")

    # Fix commands to desired values
    command_cfg["command_ranges"]["lin_vel_x"]  = [args.vel,  args.vel]
    command_cfg["command_ranges"]["lin_vel_y"]  = [args.yvel, args.yvel]
    command_cfg["command_ranges"]["ang_vel_yaw"]= [args.rot,  args.rot]


    # Create environment with single env for detailed inspection
    env = DodoEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load the trained model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_name = f"model_{args.ckpt}.pt" if args.ckpt >= 0 else "model_final.pt"
    print(f"Loading checkpoint: {ckpt_name}")
    runner.load(os.path.join(log_dir, ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)

    print_separator("DODO STEP-BY-STEP EVALUATION")
    print(f"Target velocity: [{args.vel:.2f}, {args.yvel:.2f}] m/s, angular: {args.rot:.2f} rad/s")
    print(f"Max steps: {args.max_steps}")
    print(f"Press Enter to advance each step, Ctrl+C to exit")

    # Reset environment
    obs, _ = env.reset()
    step = 0
    
    with torch.no_grad():
        try:
            while step < args.max_steps:
                # Wait for user input
                input(f"\n[Step {step:4d}] Press Enter to advance simulation...")
                
                # Get action from policy
                actions = policy(obs)
                
                # Step environment
                obs, rewards, dones, infos = env.step(actions)
                
                # Print detailed information
                print_key_parameters(env, step)
                print_reward_values(env, step)
                
                # Print episode info
                print(f"\nEpisode Info:")
                print(f"  episode_length: {env.episode_length_buf[0].item()}")
                print(f"  episode_reward: {rewards[0].item():.4f}")
                print(f"  done: {dones[0].item()}")
                
                # Check termination conditions
                if env.episode_length_buf[0] > env.max_episode_length:
                    print(f"  -> Episode terminated: max length reached")
                if torch.abs(env.base_euler[0,1]) > env.env_cfg["termination_if_pitch_greater_than"]:
                    print(f"  -> Episode terminated: pitch too large ({env.base_euler[0,1]:.1f}°)")
                if torch.abs(env.base_euler[0,0]) > env.env_cfg["termination_if_roll_greater_than"]:
                    print(f"  -> Episode terminated: roll too large ({env.base_euler[0,0]:.1f}°)")
                
                step += 1
                
                # Reset if episode done
                if dones[0]:
                    print("\n" + "="*50)
                    print("Episode finished! Resetting...")
                    print("="*50)
                    obs, _ = env.reset()
                    step = 0
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")

    print_separator("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()