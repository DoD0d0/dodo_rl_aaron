import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Dodoenv1:
    """
    DodoEnv: Go2-style biped environment adapted for the Dodo robot (8 DoF).
    Implements full omnidirectional commands (vx, vy, yaw), reward functions,
    and the RSL-RL environment interface expected by OnPolicyRunner.
    """

    CONTACT_HEIGHT = 0.05
    SWING_HEIGHT_THRESHOLD = 0.10

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # basic config
        self.num_envs = num_envs
        self.device = gs.device
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_commands = command_cfg["num_commands"]

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)

        # observation / reward scalings
        self.obs_scales = obs_cfg.get("obs_scales", {})
        self.reward_scales = reward_cfg.get("reward_scales", {})

        # ---------------- Scene & robot ----------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(1, self.num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=64,
            ),
            show_viewer=show_viewer,
        )
        # plane
        try:
            self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        except Exception:
            self.scene.add_entity(gs.morphs.Plane(fixed=True))

        # base pose
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # robot URDF
        robot_path = self.env_cfg.get("robot_mjcf", "/Users/aaronalexander/DoDodo/dodobot_v3/urdf/dodobot_v3.urdf")
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=robot_path,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        # build scene with n_envs
        self.scene.build(n_envs=num_envs)

        # ---- joints -> motors dof indices ----
        # handle joints that may have >1 DOF safely
        self.motors_dof_idx = []
        for name in self.env_cfg["joint_names"]:
            joint = self.robot.get_joint(name)
            dof_start = getattr(joint, "dof_start", None)
            dof_count = getattr(joint, "dof_count", getattr(joint, "num_dofs", 1))
            if dof_start is None:
                raise RuntimeError(f"Joint '{name}' doesn't expose dof_start; check URDF or Genesis API.")
            for i in range(int(dof_start), int(dof_start) + int(dof_count)):
                self.motors_dof_idx.append(i)

        # PD gains
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # force range/clipping if API available (optional)
        try:
            lower = (-self.env_cfg.get("clip_actions", 100.0)) * torch.ones(self.num_actions, dtype=torch.float32)
            upper = (self.env_cfg.get("clip_actions", 100.0)) * torch.ones(self.num_actions, dtype=torch.float32)
            self.robot.set_dofs_force_range(lower=lower, upper=upper, dofs_idx_local=self.motors_dof_idx)
        except Exception:
            pass

        # ankle links for foot height measures
        self.ankle_links = []
        for ln in self.env_cfg.get("foot_link_names", []):
            try:
                self.ankle_links.append(self.robot.get_link(ln))
            except Exception:
                # ignore if link not found
                pass

        # indices for convenience
        self.hip_aa_indices = []
        self.hip_fe_indices = []
        self.knee_fe_indices = []
        names = self.env_cfg["joint_names"]
        # try to find canonical names used in provided configs
        for key in ("left_joint_1", "left_joint_2", "left_joint_3"):
            if key in names:
                self.hip_aa_indices.append(names.index("left_joint_1"))
                self.hip_fe_indices.append(names.index("left_joint_2"))
                self.knee_fe_indices.append(names.index("left_joint_3"))
                break
        # fallback: split by leg halves if possible
        if len(self.hip_aa_indices) == 0 and len(names) >= 6:
            # assume order L1,R1,L2,R2,L3,R3,L4,R4 etc — user-provided joint_names should be accurate
            pass

        # ---------------- Buffers ----------------
        N, A, C = self.num_envs, self.num_actions, self.num_commands
        self.base_lin_vel = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).repeat(N, 1)

        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((N,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((N,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((N,), device=self.device, dtype=torch.int32)
        self.commands = torch.zeros((N, C), device=self.device, dtype=torch.float32)
        self.commands_scale = torch.tensor(
            [self.obs_scales.get("lin_vel", 1.0), self.obs_scales.get("lin_vel", 1.0), self.obs_scales.get("ang_vel", 1.0)],
            device=self.device,
            dtype=torch.float32,
        )
        self.actions = torch.zeros((N, A), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.base_quat = torch.zeros((N, 4), device=self.device, dtype=torch.float32)
        self.base_euler = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.current_ankle_heights = torch.zeros((N, max(1, len(self.ankle_links))), device=self.device, dtype=torch.float32)
        self.prev_contact = torch.zeros((N, max(1, len(self.ankle_links))), device=self.device, dtype=torch.float32)

        # default dof positions as tensor
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float32,
        )

        # extras expected by runner
        self.extras = {"observations": {"critic": None}, "time_outs": torch.zeros((N,), device=self.device)}
    
        # ---------- reward functions binding & episode_sums ----------
        # multiply reward scales by dt (so scales are per second -> per step)
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            # keep original scale but convert to per-step
            self.reward_scales[name] = float(self.reward_scales[name]) * self.dt
            fn = getattr(self, "_reward_" + name, None)
            if fn is None:
                raise AttributeError(f"Reward function _reward_{name} missing in DodoEnv.")
            self.reward_functions[name] = fn
            self.episode_sums[name] = torch.zeros((N,), device=self.device, dtype=torch.float32)

        # initialize commands (random initial commands)
        self._resample_commands(torch.arange(N, device=self.device))

    # ---------------- command resampling ----------------
    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # command_cfg may provide either ranges or single values; support both
        lx = self.command_cfg.get("lin_vel_x_range", [0.0, 0.0])
        ly = self.command_cfg.get("lin_vel_y_range", [0.0, 0.0])
        ay = self.command_cfg.get("ang_vel_range", [0.0, 0.0])
        self.commands[envs_idx, 0] = gs_rand_float(lx[0], lx[1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(ly[0], ly[1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(ay[0], ay[1], (len(envs_idx),), self.device)

    # ---------------- step / simulation ----------------
    def step(self, actions):
        # clamp actions
        self.actions = torch.clip(actions, -self.env_cfg.get("clip_actions", 100.0), self.env_cfg.get("clip_actions", 100.0))

        # simulate latency if requested
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # convert actions -> target positions (policy outputs deltas around default)
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # send to Genesis actuator API (position control)
        try:
            self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        except Exception:
            # fall back to numpy if needed
            self.robot.control_dofs_position(target_dof_pos.cpu().numpy(), self.motors_dof_idx)

        # step physics
        self.scene.step()

        # ---- update buffers ----
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        # base_euler in radians (rpy)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=False,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)

        # read dof states
        try:
            self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        except Exception:
            # fallback: read full vectors then index
            self.dof_pos[:] = self.robot.get_dofs_position()[..., self.motors_dof_idx]
            self.dof_vel[:] = self.robot.get_dofs_velocity()[..., self.motors_dof_idx]

        # ankle heights
        if len(self.ankle_links) > 0:
            try:
                self.current_ankle_heights[:] = torch.stack([link.get_pos()[:, 2] for link in self.ankle_links], dim=1)
            except Exception:
                # if link.get_pos isn't batched, call per-env and stack -- skip to avoid slowdown
                pass

        # resample commands periodically
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (self.episode_length_buf % resample_every == 0).nonzero(as_tuple=False).reshape((-1,))
        if envs_idx.numel() > 0:
            self._resample_commands(envs_idx)

        # termination checks
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"] * math.pi / 180.0
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"] * math.pi / 180.0

        # mark timeouts for extras
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float32)
        if time_out_idx.numel() > 0:
            self.extras["time_outs"][time_out_idx] = 1.0

        # reset requested envs (collect stats before clearing)
        reset_idx = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if reset_idx.numel() > 0:
            self.reset_idx(reset_idx)

        # ---- compute rewards ----
        self.rew_buf[:] = 0.0
        for name, reward_fn in self.reward_functions.items():
            r = reward_fn() * self.reward_scales[name]
            self.rew_buf += r
            self.episode_sums[name] += r

        # ---- observations ----
        # build obs vector: base_ang_vel(3), projected_gravity(3), commands(3), dof_pos- default (A), dof_vel (A), actions (A)
        obs_parts = [
            self.base_ang_vel * self.obs_scales.get("ang_vel", 1.0),
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.get("dof_pos", 1.0),
            self.dof_vel * self.obs_scales.get("dof_vel", 1.0),
            self.actions,
        ]
        self.obs_buf = torch.cat(obs_parts, dim=-1)

        # extras for critic
        self.extras["observations"]["critic"] = self.obs_buf.clone()
        # per-step episode rewards for logging
        ep_step = {k: v.clone() for k, v in self.episode_sums.items()}
        self.extras["episode"] = ep_step

        # bookkeeping
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ---------------- get_observations required by runner ----------------
    def get_observations(self):
        # ensure extras and critic exist
        self.extras["observations"]["critic"] = self.obs_buf.clone()
        return self.obs_buf, self.extras

    # ---------------- reset subset of envs ----------------
    def reset_idx(self, envs_idx):
        if envs_idx.numel() == 0:
            return
        # reset physics state for these envs
        try:
            self.scene.reset(env_ids=envs_idx)
        except Exception:
            # if Scene.reset not available, fall back to setting DOFs/pos directly
            pass

        # reset DOFs
        self.dof_pos[envs_idx] = self.default_dof_pos.unsqueeze(0).repeat(len(envs_idx), 1)
        self.dof_vel[envs_idx] = 0.0
        try:
            self.robot.set_dofs_position(
                position=self.dof_pos[envs_idx],
                dofs_idx_local=self.motors_dof_idx,
                zero_velocity=True,
                envs_idx=envs_idx,
            )
        except Exception:
            try:
                self.robot.set_qpos(self.dof_pos[envs_idx].cpu().numpy(), env_ids=envs_idx)
            except Exception:
                pass

        # reset base pose
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        try:
            self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
            self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        except Exception:
            pass

        # reset velocities
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        try:
            self.robot.zero_all_dofs_velocity(envs_idx)
        except Exception:
            pass

        # reset other buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras episode averages and clear episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / max(1.0, self.env_cfg["episode_length_s"])
            )
            self.episode_sums[key][envs_idx] = 0.0

        # resample commands for these envs
        self._resample_commands(envs_idx)

    # ---------------- reset all ----------------
    def reset(self):
        self.reset_buf[:] = True
        idx = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(idx)
        # build an initial observation
        obs = torch.cat(
            [
                self.base_ang_vel * self.obs_scales.get("ang_vel", 1.0),
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.get("dof_pos", 1.0),
                self.dof_vel * self.obs_scales.get("dof_vel", 1.0),
                self.actions,
            ],
            dim=-1,
        )
        self.obs_buf = obs
        self.extras["observations"]["critic"] = obs.clone()
        return self.obs_buf, None

    # ---------------- reward functions ----------------
    def _reward_periodic_gait(self):
        phase = (self.episode_length_buf.float() * self.dt) % self.reward_cfg.get("period", 1.0)
        half = self.reward_cfg.get("period", 1.0) * 0.5
        contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float() if self.current_ankle_heights.numel() else torch.zeros((self.num_envs, 1), device=self.device)
        desired_left = (phase < half).float()
        desired_right = (phase >= half).float()
        # if only one foot link, treat indexes carefully
        if contact.shape[1] == 1:
            return (desired_left * contact[:, 0] + desired_right * contact[:, 0]).clamp(0.0, 1.0)
        return (desired_left * contact[:, 0] + desired_right * contact[:, 1]).clamp(0.0, 1.0)

    def _reward_energy_penalty(self):
        err = torch.sum((self.actions - self.last_actions) ** 2, dim=1)
        sigma = self.reward_cfg.get("energy_sigma", 1.0)
        return torch.exp(-err / (2 * sigma ** 2))

    def _reward_foot_swing_clearance(self):
        hs = self.current_ankle_heights if self.current_ankle_heights.numel() else torch.zeros((self.num_envs, 1), device=self.device)
        contact = (hs < self.CONTACT_HEIGHT).float()
        swing_mask = 1.0 - contact
        clearance = hs * swing_mask
        excess = torch.relu(clearance - self.reward_cfg.get("clearance_target", 0.15))
        return torch.mean(excess, dim=1)

    def _reward_forward_torso_pitch(self):
        pitch = self.base_euler[:, 1]  # radians
        err = (pitch - self.reward_cfg.get("pitch_target", 0.0)) ** 2
        sigma = self.reward_cfg.get("pitch_sigma", 0.1)
        return torch.exp(-err / (2 * sigma ** 2))

    def _reward_knee_extension_at_push(self):
        # reward knee extension during stance; indices depend on joint naming
        try:
            idx_l = self.env_cfg["joint_names"].index("left_joint_3")
            idx_r = self.env_cfg["joint_names"].index("right_joint_3")
            hs = self.current_ankle_heights
            stance = (hs < self.CONTACT_HEIGHT).any(dim=1).float()
            ext_l = 1.0 - torch.relu(self.dof_pos[:, idx_l])
            ext_r = 1.0 - torch.relu(-self.dof_pos[:, idx_r])
            return stance * ((ext_l + ext_r) * 0.5)
        except Exception:
            return torch.zeros((self.num_envs,), device=self.device)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=1)
        return torch.exp(-lin_vel_error / max(1e-6, self.reward_cfg.get("tracking_sigma", 0.25)))

    def _reward_tracking_ang_vel(self):
        ang_vel_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        return torch.exp(-ang_vel_error / max(1e-6, self.reward_cfg.get("tracking_sigma", 0.25)))

    def _reward_lateral_drift_penalty(self):
        drift = self.base_lin_vel[:, 1].abs()
        sigma = self.reward_cfg.get("drift_sigma", 0.1)
        return torch.exp(-drift ** 2 / (2 * sigma ** 2))

    def _reward_hip_abduction_penalty(self):
        try:
            idx_l = self.env_cfg["joint_names"].index("left_joint_1")
            idx_r = self.env_cfg["joint_names"].index("right_joint_1")
            abd_l = self.dof_pos[:, idx_l]
            abd_r = self.dof_pos[:, idx_r]
            err = abd_l ** 2 + abd_r ** 2
            sigma = self.reward_cfg.get("hip_abduction_sigma", 0.1)
            return torch.exp(-err / (2 * sigma ** 2))
        except Exception:
            return torch.zeros((self.num_envs,), device=self.device)

    def _reward_orientation_stability(self):
        roll = self.base_euler[:, 0]
        pitch = self.base_euler[:, 1]
        err = roll ** 2 + pitch ** 2
        sigma = self.reward_cfg.get("orient_sigma", 0.1)
        return torch.exp(-err / (2 * sigma ** 2))

    def _reward_base_height(self):
        err = (self.base_pos[:, 2] - self.reward_cfg.get("base_height_target", 0.35)) ** 2
        sigma = self.reward_cfg.get("height_sigma", 0.1)
        return torch.exp(-err / (2 * sigma ** 2))

    def _reward_survive(self):
        done = self.reset_buf.float()  # 1.0 when env ended
        return (1.0 - done)

    def _reward_fall_penalty(self):
        roll = self.base_euler[:, 0]
        pitch = self.base_euler[:, 1]
        thr_r = self.reward_cfg.get("roll_threshold", 30 * math.pi / 180.0)
        thr_p = self.reward_cfg.get("pitch_threshold", 30 * math.pi / 180.0)
        mask = ((roll.abs() > thr_r) | (pitch.abs() > thr_p)).float()
        return -mask

    def _reward_action_rate(self):
        return torch.sum((self.last_actions - self.actions) ** 2, dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_bird_hip_phase(self):
        """
        Vogel‑typischer Hüft‑FE‑Zyklustreiber als Gauß‑Reward.
        """
        idx_l = self.env_cfg["joint_names"].index("left_joint_2")
        idx_r = self.env_cfg["joint_names"].index("right_joint_2")
        phase = ((self.episode_length_buf.float() * self.dt) % self.reward_cfg["period"]) / self.reward_cfg["period"]
        omega = 2 * math.pi * phase
        tgt  = self.reward_cfg["bird_hip_target"]
        amp  = self.reward_cfg["bird_hip_amp"]
        desired_l = tgt + amp * torch.sin(omega)
        desired_r = tgt - amp * torch.sin(omega)
        a_l = self.dof_pos[:, idx_l]
        a_r = self.dof_pos[:, idx_r]
        err = (a_l - desired_l)**2 + (a_r - desired_r)**2
        sigma = self.reward_cfg["bird_hip_sigma"]
        return torch.exp(-err / (2 * sigma**2))