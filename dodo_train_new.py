import argparse
import os
import pickle
import torch
import shutil
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import genesis as gs
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from dodo_env_new_2 import Dodoenvironment
import wandb
import copy

# -----------------------------------------------------------------------------
# Global logs (all important reward terms)
# -----------------------------------------------------------------------------
iters = []
val_loss = []
surrogate_loss = []
noise_std = []
total_reward = []
ep_length = []

# Reward specific logs
tracking_lin_vel = []
tracking_ang_vel = []
base_height = []
fall_penalty = []
action_rate = []  

#inacive reward terms
periodic_gait = []
energy_penalty = []
foot_swing_clearance = []
forward_torso_pitch = []
knee_extension_at_push = []
bird_hip_phase = []
hip_abduction_penalty = []
lateral_drift_penalty = []

def wandb_log(step, stats):
    print(f"[WandB] Iter {step} | reward={stats['episode_reward_mean']:.2f} | loss={stats['value_loss']:.4f}")
    wandb.log(stats, step=step)

def log_and_plot(log_dir, it, stats):
    iters.append(it)
    val_loss.append(stats["value_loss"])
    surrogate_loss.append(stats["surrogate_loss"])
    noise_std.append(stats["action_noise_std"])
    total_reward.append(stats["episode_reward_mean"])
    ep_length.append(stats["episode_length_mean"])

    # Active Rewards
    tracking_lin_vel.append(stats.get("tracking_lin_vel", 0.0))
    tracking_ang_vel.append(stats.get("tracking_ang_vel", 0.0))
    base_height.append(stats.get("base_height", 0.0))
    fall_penalty.append(stats.get("fall_penalty", 0.0))
    action_rate.append(stats.get("action_rate", 0.0)) 

    # Inactive Rewards
    periodic_gait.append(stats.get("periodic_gait", 0.0))
    energy_penalty.append(stats.get("energy_penalty", 0.0))
    foot_swing_clearance.append(stats.get("foot_swing_clearance", 0.0))
    forward_torso_pitch.append(stats.get("forward_torso_pitch", 0.0))
    knee_extension_at_push.append(stats.get("knee_extension_at_push", 0.0))
    bird_hip_phase.append(stats.get("bird_hip_phase", 0.0))
    hip_abduction_penalty.append(stats.get("hip_abduction_penalty", 0.0))
    lateral_drift_penalty.append(stats.get("lateral_drift_penalty", 0.0))

    wandb_log(it, stats)

    if it % 100 == 0:
        fig, axes = plt.subplots(3, 5, figsize=(24, 12))
        axes = axes.flatten()
        metrics = [
            val_loss, surrogate_loss, noise_std,
            total_reward, ep_length, tracking_lin_vel, tracking_ang_vel, 
            base_height, fall_penalty, action_rate,
            periodic_gait, energy_penalty, foot_swing_clearance,
            forward_torso_pitch, knee_extension_at_push,
            bird_hip_phase, hip_abduction_penalty, lateral_drift_penalty,
        ]
        titles = [
            "Value Loss", "Surrogate Loss", "Action Noise Std",
            "Mean Total Reward", "Mean Episode Length", "Tracking Lin Vel", 
            "Tracking Ang Vel", "Base Height",
            "Fall Penalty", "Action Rate Penalty",
            "Periodic Gait", "Energy Penalty", "Foot Swing Clearance",
            "Forward Torso Pitch", "Knee Ext. at Push",
            "Bird Hip Phase", "Hip Abduction Penalty", "Lateral Drift"
        ]
        for ax, metric, title in zip(axes, metrics, titles):
            ax.plot(iters, metric)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        save_path = os.path.join(log_dir, "metrics.png")
        fig.savefig(save_path)
        wandb.log({"metrics_plot": wandb.Image(save_path)}, step=it)
        plt.close(fig)
        print(f"[Plot] saved to {save_path}")

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005, #was 0.02
            "gamma": 0.98,
            "lam": 0.95,
            "learning_rate": 0.0002,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "fixed",    #was adaptive
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
        "logger": "wandb",
        "tensorboard_subdir": "tb",
    }

def get_cfgs():
    env_cfg = {
        "num_actions": 8,
        "default_joint_angles": {
            "left_joint_1": 0.0, "right_joint_1": 0.0,
            "left_joint_2": 0.4, "right_joint_2": 0.4,
            "left_joint_3": -0.7, "right_joint_3": -0.7,
            "left_joint_4": 0.3, "right_joint_4": 0.3
        },
        "joint_names": [
            "left_joint_1", "right_joint_1",
            "left_joint_2", "right_joint_2",
            "left_joint_3", "right_joint_3",
            "left_joint_4", "right_joint_4"
        ],
        "kp": 40.0,
        "kd": 2.0 * math.sqrt(40.0),
        "termination_if_roll_greater_than": 30,
        "termination_if_pitch_greater_than": 30,
        "base_init_pos": [0.0, 0.0, 0.55],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 10.0,
        "resampling_time_s": 2.0,
        "action_scale": 0.25,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
        "robot_mjcf": "/Users/aaronalexander/DoDodo/dodobot_v3/urdf/dodobot_v3_simple.urdf",
        "foot_link_names": ["left_link_4", "right_link_4"]
    }
    obs_cfg = {
        # "num_obs": 6(vel and ang vel) + 3(projected gravity) * [env_cfg["num_actions"] + 3](dof_pos, dof_vel, actions) + 3(commands) = 36
        "num_obs": 36,  #added this 27.11, now includes lin vel, ang vel, proj gravity, commands, dof pos, dof vel, and previous action
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.5,
            #"orientation_stability": 0.3,
            "base_height": 1.0,
            #"survive": 0.0,
            "fall_penalty": 10.0,
            #"periodic_gait": 0.0,
            "foot_swing_clearance": -5.0,
            #"knee_extension_at_push": 0.0,
            #"bird_hip_phase": 0.0,
            #"forward_torso_pitch": 0.0,
            #"hip_abduction_penalty": 0.0,
            #"lateral_drift_penalty": 0.0,
            #"energy_penalty": -0.01,
            "action_rate": -0.1
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.35,
        "height_sigma": 0.10,
        "orient_sigma": 0.10,
        "energy_sigma": 1.00,
        "period": 1.00,
        "clearance_target": 0.15,
        "pitch_target": 0.17,
        "pitch_sigma": 0.10,
        "bird_hip_target": -0.35,
        "bird_hip_amp": 0.15,
        "bird_hip_sigma": 0.10,
        "hip_abduction_sigma": 0.10,
        "drift_sigma": 0.10,
        "pitch_threshold": 30 * math.pi/180,
        "roll_threshold": 30 * math.pi/180,
    }

    command_cfg = {
        "num_commands": 3,
        "resampling_time_s": 2.0,
        "command_ranges": {
            "lin_vel_x": [0.1, 0.3],
            "lin_vel_y": [ -0.1, 0.1],
            "ang_vel_yaw": [-0.3, 0.3]
        }       
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking-new")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500)
    args = parser.parse_args()

    wandb.init(project="dodo-birdlike-gait-new", name=args.exp_name)
    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    wandb.config.update({
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "env_cfg": env_cfg,
        "reward_scales": reward_cfg["reward_scales"],
        "obs_cfg": obs_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
    })

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    class CustomRunner(OnPolicyRunner):
        def __init__(self, env, train_cfg, log_dir, device):
            super().__init__(env, train_cfg, log_dir, device)
            self.log_dir = log_dir

        def save(self, path):
            checkpoint = {
                "model_state_dict": self.alg.actor_critic.state_dict(), 
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": getattr(self, "infos", {}),
            }
            if hasattr(self.alg, "lr_scheduler"):
                checkpoint["scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
            torch.save(checkpoint, path)
            print(f"[CustomRunner] ✅ Saved checkpoint to {path}")

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            self.env.reset()
            obs, extras = self.env.get_observations()
            critic_obs = extras["observations"]["critic"].to(self.device)
            obs = obs.to(self.device)
            self.train_mode()

            for it in range(self.current_learning_iteration, num_learning_iterations):
                ep_infos, rewbuffer, lenbuffer = [], [], []

                # Rollout collect
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    obs = self.obs_normalizer(obs)
                    critic_obs = infos["observations"]["critic"].to(self.device)
                    critic_obs = self.critic_obs_normalizer(critic_obs)

                    self.alg.process_env_step(rewards, dones, infos)
                    ep_infos.append(infos["episode"])
                    rewbuffer.append(rewards.mean().item())
                    lenbuffer.append((~dones).sum().item())

                # Update
                self.alg.compute_returns(critic_obs)
                mv, ms, *_ = self.alg.update()

                # Stats init
                stats = {
                    "value_loss": mv,
                    "surrogate_loss": ms,
                    "action_noise_std": self.alg.actor_critic.action_std.mean().item(),
                    "episode_reward_mean": np.mean(rewbuffer),
                    "episode_length_mean": np.mean(lenbuffer),
                }



                # --- FIXED LOGIC START ---
                for name in self.env.reward_scales.keys():
                    stats[name] = 0.0

                mean_logs = {}
                for ep in ep_infos:
                    for k, v in ep.items():
                        # Strip "rew_" to match stats keys
                        clean_name = k.replace("rew_", "")
                        
                        if clean_name in stats:
                            mean_logs.setdefault(clean_name, []).append(v)
                
                for k, v_list in mean_logs.items():
                    stats[k] = float(np.mean(v_list))
                # --- FIXED LOGIC END ---

                log_and_plot(self.log_dir, it, stats)
                self.current_learning_iteration = it

                # --- improper logic that was causing 0.0 for all reward terms ---
                # # zero all reward terms then fill from ep_infos
                # for name in self.env.reward_scales.keys():
                #     stats[name] = 0.0

                # mean_logs = {}
                # for ep in ep_infos:
                #     for k, v in ep.items():
                #         if k in stats:
                #             mean_logs.setdefault(k, []).append(v.mean().cpu().item())
                # for k, v_list in mean_logs.items():
                #     stats[k] = float(np.mean(v_list))
                # --- end of improper logic that was causing 0.0 for all reward terms ---



    # Create env and runner, then train
    env = Dodoenvironment(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False
    )
    env.reset()

    runner = CustomRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))

    print(f"=== Trained model saved at {log_dir}/model_final.pt ===")

if __name__ == "__main__":
    main()
