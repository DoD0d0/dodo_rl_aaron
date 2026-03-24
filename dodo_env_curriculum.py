import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Dodoenvironment:
    CONTACT_HEIGHT = 0.05
    
    # --- SCANNER CONFIG ---
    # 9x9 grid spanning 1.6m x 1.6m around the robot
    # This acts as the robot's "eyes" to see bumps and holes
    SCAN_X_OFFSETS = np.linspace(-0.8, 0.8, 9)
    SCAN_Y_OFFSETS = np.linspace(-0.8, 0.8, 9)
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
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

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ---------------------------------------------------------
        # 1. SCENE & TERRAIN GENERATION (Curriculum)
        # ---------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # --- TERRAIN CONFIG ---
        self.n_rows = 10  # Difficulty Levels (0 = Easy, 9 = Hard)
        self.n_cols = 20  # Width of the world
        self.subterrain_width = 4.0 # Width of one lane
        self.terrain_length = 20.0  # Length of the strip
        self.horizontal_scale = 0.1 # 10cm resolution
        self.vertical_scale = 0.005 # 5mm height resolution
        
        # Generate Height Field
        grid_rows = int(self.n_rows * self.subterrain_width / self.horizontal_scale)
        grid_cols = int(self.terrain_length / self.horizontal_scale)
        self.height_field_raw = np.zeros((grid_rows, grid_cols), dtype=np.int16)
        
        for row in range(self.n_rows):
            # Calculate difficulty (0.0 to 1.0)
            difficulty = row / (self.n_rows - 1)
            
            start_x = int(row * self.subterrain_width / self.horizontal_scale)
            end_x = int((row + 1) * self.subterrain_width / self.horizontal_scale)
            
            # Generate random bumps
            # We use uniform noise, but Perlin noise is better for "smooth" hills.
            # Scaling: Max bump height = 0.35m * difficulty
            noise = np.random.uniform(-1.0, 1.0, (end_x - start_x, grid_cols))
            scaled_noise = noise * difficulty * 0.35 / self.vertical_scale
            self.height_field_raw[start_x:end_x, :] = scaled_noise.astype(np.int16)

        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                height_field=self.height_field_raw,
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
                pos=(0.0, 0.0, 0.0)
            )
        )
        
        # Move terrain data to GPU for the scanner
        self.height_samples = torch.tensor(self.height_field_raw, device=self.device).float() * self.vertical_scale
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # ---------------------------------------------------------
        # 2. SCANNER SETUP
        # ---------------------------------------------------------
        scan_x, scan_y = np.meshgrid(self.SCAN_X_OFFSETS, self.SCAN_Y_OFFSETS)
        self.scan_points = torch.tensor(np.stack([scan_x.flatten(), scan_y.flatten()], axis=1), device=self.device, dtype=gs.tc_float)
        self.num_scan_points = self.scan_points.shape[0] # 81 points

        # ---------------------------------------------------------
        # 3. ROBOT SETUP
        # ---------------------------------------------------------
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg.get("robot_mjcf", "dodobot_v3/urdf/dodobot_v3_simple.urdf"),
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        self.scene.build(n_envs=num_envs)

        # Buffers & Indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.ankle_links = []
        for ln in self.env_cfg.get("foot_link_names", []):
            try: self.ankle_links.append(self.robot.get_link(ln))
            except: pass

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.episode_lin_vel_sum = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.current_ankle_heights = torch.zeros((self.num_envs, max(1, len(self.ankle_links))), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], device=gs.device, dtype=gs.tc_float)
        self.extras = dict(); self.extras["observations"] = dict()
        self._resample_commands(torch.arange(self.num_envs, device=gs.device))

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0: return
        ranges = self.command_cfg.get("command_ranges", {})
        lx = ranges.get("lin_vel_x", [0.0, 0.0])
        ly = ranges.get("lin_vel_y", [0.0, 0.0])
        ay = ranges.get("ang_vel_yaw", [0.0, 0.0])
        self.commands[envs_idx, 0] = gs_rand_float(lx[0], lx[1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(ly[0], ly[1], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(ay[0], ay[1], (len(envs_idx),), self.device)

    def _get_height_scan(self):
        """ 
        Efficiently samples terrain height at 81 points around each robot 
        Returns: Heights relative to the robot's base height
        """
        # Expand scan points: (N, 81, 2)
        points = self.scan_points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        # Create 3D points for rotation (x, y, 0)
        points_3d = torch.zeros((self.num_envs, self.num_scan_points, 3), device=self.device)
        points_3d[:, :, :2] = points
        
        # Rotate scan grid to match robot's heading (Yaw)
        # We transform by the robot's quaternion
        flat_points = points_3d.view(-1, 3)
        flat_quat = self.base_quat.unsqueeze(1).repeat(1, self.num_scan_points, 1).view(-1, 4)
        rot_points = transform_by_quat(flat_points, flat_quat).view(self.num_envs, self.num_scan_points, 3)
        
        # Add Robot Position to get World Coordinates
        scan_x = rot_points[:, :, 0] + self.base_pos[:, 0].unsqueeze(1)
        scan_y = rot_points[:, :, 1] + self.base_pos[:, 1].unsqueeze(1)
        
        # Convert World Coords -> Grid Indices
        scan_x = torch.clamp(scan_x, 0, (self.n_rows * self.subterrain_width) - 0.1)
        scan_y = torch.clamp(scan_y, -self.terrain_length/2, self.terrain_length/2 - 0.1)
        
        idx_x = (scan_x / self.horizontal_scale).long()
        idx_y = ((scan_y + self.terrain_length/2) / self.horizontal_scale).long()
        
        idx_x = torch.clamp(idx_x, 0, self.height_samples.shape[0]-1)
        idx_y = torch.clamp(idx_y, 0, self.height_samples.shape[1]-1)
        
        # Sample heights and subtract robot base height
        heights = self.height_samples[idx_x, idx_y] # (N, 81)
        return heights - self.base_pos[:, 2].unsqueeze(1)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()
        
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.episode_lin_vel_sum += self.base_lin_vel[:, 0]
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        if len(self.ankle_links) > 0:
            try: self.current_ankle_heights[:] = torch.stack([link.get_pos()[:, 2] for link in self.ankle_links], dim=1)
            except: pass

        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).reshape((-1,)))
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name] * self.dt
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # --- GET EXTREOCEPTION ---
        height_scan = self._get_height_scan() # Shape: (N, 81)

        # Compute Observations (Now with scan!)
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 8
                self.dof_vel * self.obs_scales["dof_vel"],  # 8
                self.actions,  # 8
                height_scan, # 81 (New!)
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    def get_privileged_observations(self): return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0: return

        # --- CURRICULUM LOGIC ---
        # 1. Survive full duration = Move Up
        move_up = (self.episode_length_buf[envs_idx] >= self.max_episode_length)
        # 2. Die early (<50%) = Move Down
        move_down = (self.episode_length_buf[envs_idx] < self.max_episode_length * 0.5)

        self.terrain_levels[envs_idx] += move_up.long()
        self.terrain_levels[envs_idx] -= move_down.long()
        self.terrain_levels[envs_idx] = torch.clamp(self.terrain_levels[envs_idx], 0, self.n_rows - 1)

        # Respawn Logic
        # Calculate X position based on assigned level (Row)
        new_x = (self.terrain_levels[envs_idx] * self.subterrain_width) + (self.subterrain_width / 2.0)
        # Random Y along the strip
        new_y = (torch.rand(len(envs_idx), device=self.device) * self.terrain_length) - (self.terrain_length/2)
        
        # Get Ground Height at spawn point so we don't spawn under the floor
        idx_x = (new_x / self.horizontal_scale).long()
        idx_y = ((new_y + self.terrain_length/2) / self.horizontal_scale).long()
        idx_x = torch.clamp(idx_x, 0, self.height_samples.shape[0]-1)
        idx_y = torch.clamp(idx_y, 0, self.height_samples.shape[1]-1)
        ground_z = self.height_samples[idx_x, idx_y]
        
        self.base_pos[envs_idx, 0] = new_x
        self.base_pos[envs_idx, 1] = new_y
        self.base_pos[envs_idx, 2] = ground_z + 0.55 
        
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_init_quat.repeat(len(envs_idx), 1), zero_velocity=False, envs_idx=envs_idx)
        
        # Reset Stats
        episode_len = self.episode_length_buf[envs_idx].float()
        avg_vel = self.episode_lin_vel_sum[envs_idx] / episode_len
        is_success = (self.episode_length_buf[envs_idx] >= self.max_episode_length).float()
        is_failure = 1.0 - is_success
        
        if "episode" not in self.extras: self.extras["episode"] = {}
        self.extras["episode"]["mean_lin_vel_x"] = torch.mean(avg_vel).item()
        self.extras["episode"]["success_rate"] = torch.mean(is_success).item()
        self.extras["episode"]["fall_rate"] = torch.mean(is_failure).item()
        
        # Log the curriculum level so you can see it on WandB!
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels[envs_idx].float()).item()
        
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"])
            self.episode_sums[key][envs_idx] = 0.0

        self.episode_lin_vel_sum[envs_idx] = 0.0
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # --- REWARD FUNCTIONS ---
    # ============================================================
    # 1. Linear velocity tracking
    # ============================================================
    def _reward_tracking_lin_vel(self):
        error = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=1)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-error / (2 * sigma ** 2)).clamp(0.0, 1.0)



    # ============================================================
    # 2. Angular velocity tracking
    # ============================================================
    def _reward_tracking_ang_vel(self):
        error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-error / (2 * sigma ** 2)).clamp(0.0, 1.0)


    # ============================================================
    # 3. Orientation stability (upright)
    # ============================================================
    def _reward_orientation_stability(self):
        # base_euler is in DEGREES!
        roll_deg = self.base_euler[:, 0]
        pitch_deg = self.base_euler[:, 1]
        # Convert to radians for computation
        roll_rad = roll_deg * math.pi / 180.0
        pitch_rad = pitch_deg * math.pi / 180.0
        err = roll_rad**2 + pitch_rad**2
        sigma = self.reward_cfg.get("orient_sigma", 0.1)
        return torch.exp(-err / (2 * sigma**2))

    # ============================================================
    # 4. Base height
    # ============================================================
    def _reward_base_height(self):
        target = self.reward_cfg.get("base_height_target", 0.35)
        err = (self.base_pos[:, 2] - target)**2
        sigma = self.reward_cfg.get("height_sigma", 0.1)
        return torch.exp(-err / (2 * sigma**2))


    # ============================================================
    # 5. Survive (positive if not fallen)
    # ============================================================
    def _reward_survive(self):
        #return (1.0 - self.reset_buf.float()).clamp(0.0, 1.0)
        """Give constant reward for being alive and upright"""  #added 27.11
        roll = self.base_euler[:, 0].abs()
        pitch = self.base_euler[:, 1].abs()
        thr_r = self.reward_cfg.get("roll_threshold", 30.0)
        thr_p = self.reward_cfg.get("pitch_threshold", 30.0)
        upright = ((roll < thr_r) & (pitch < thr_p)).float()
        return upright

    # ============================================================
    # 6. Fall penalty (roll/pitch exceeding threshold)
    # ============================================================
    def _reward_fall_penalty(self):
        roll = self.base_euler[:, 0].abs()
        pitch = self.base_euler[:, 1].abs()

        thr_r = self.reward_cfg.get("roll_threshold", 30.0)
        thr_p = self.reward_cfg.get("pitch_threshold", 30.0)

        fail = ((roll > thr_r) | (pitch > thr_p)).float()
        return -fail


    # ============================================================
    # 7. Periodic gait reward (left stance → right stance)
    # ============================================================
    def _reward_periodic_gait(self):
        period = self.reward_cfg.get("period", 1.0)
        phase = (self.episode_length_buf.float() * self.dt) % period
        half = period * 0.5
        
        # Get contact state
        if self.current_ankle_heights.numel():
            contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        else:
            contact = torch.zeros((self.num_envs, 2), device=self.device)

        # Desired contact states (1 = ON ground, 0 = IN AIR)
        desired_left = (phase < half).float()
        desired_right = (phase >= half).float()

        # Calculate matches (1 if correct, 0 if wrong)
        # We want Left to equal Desired_Left, and Right to equal Desired_Right
        # Logic: 1.0 - abs(desired - actual)
        left_score = 1.0 - torch.abs(desired_left - contact[:, 0])
        right_score = 1.0 - torch.abs(desired_right - contact[:, 1])

        # Return average success of both legs
        return 0.5 * (left_score + right_score)

    # def _reward_periodic_gait(self):
    #     period = self.reward_cfg.get("period", 1.0)
    #     phase = (self.episode_length_buf.float() * self.dt) % period
    #     half = period * 0.5
    #     contact = torch.zeros((self.num_envs, max(1, len(self.ankle_links))), device=self.device)
    #     if self.current_ankle_heights.numel():
    #         contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
    #     desired_left = (phase < half).float()
    #     desired_right = (phase >= half).float()
    #     if contact.shape[1] == 1:
    #         return (desired_left * contact[:, 0] + desired_right * contact[:, 0]).clamp(0.0, 1.0)
    #     return (desired_left * contact[:, 0] + desired_right * contact[:, 1]).clamp(0.0, 1.0)


    # ============================================================
    # 8. Foot swing clearance (don’t drag foot)
    # ============================================================
    def _reward_foot_swing_clearance(self):
        if self.current_ankle_heights.numel():
            hs = self.current_ankle_heights
        else:
            hs = torch.zeros((self.num_envs, 1), device=self.device)
            
        # Identify feet in swing phase
        contact = (hs < self.CONTACT_HEIGHT).float()
        swing_mask = 1.0 - contact
        
        # Get the target height (e.g., 0.15m)
        target = self.reward_cfg.get("clearance_target", 0.15)
        
        # --- THE FIX ---
        # Calculate how far the foot is from the target
        # We square it so it penalizes being too LOW (dragging) 
        # AND being too HIGH (wasting energy).
        error = (hs - target) ** 2
        
        # Only apply this penalty when the foot is supposed to be in the air
        swing_error = error * swing_mask
        
        return -torch.mean(swing_error, dim=1)

    # def _reward_foot_swing_clearance(self):
    #     if self.current_ankle_heights.numel():
    #         hs = self.current_ankle_heights
    #     else:
    #         hs = torch.zeros((self.num_envs, 1), device=self.device)
    #     contact = (hs < self.CONTACT_HEIGHT).float()
    #     swing_mask = 1.0 - contact
    #     clearance = hs * swing_mask
    #     target = self.reward_cfg.get("clearance_target", 0.15)
    #     excess = torch.relu(clearance - target)
    #     return -torch.mean(excess, dim=1) # negative reward for dragging foot, changed 27.11

    # ============================================================
    # 9. Knee extension at push-off
    # ============================================================
    def _reward_knee_extension_at_push(self):
        idx_l = self.env_cfg["joint_names"].index("left_joint_3")
        idx_r = self.env_cfg["joint_names"].index("right_joint_3")
        hs = self.current_ankle_heights
        stance = ((hs < self.CONTACT_HEIGHT).any(dim=1)).float()
        ext_l = 1.0 - torch.relu(self.dof_pos[:, idx_l])
        ext_r = 1.0 - torch.relu(-self.dof_pos[:, idx_r])
        return (stance * 0.5 * (ext_l + ext_r)).clamp(0.0, 1.0)


    # ============================================================
    # 10. Bird-like hip oscillation (missing one!)
    # ============================================================
    def _reward_bird_hip_phase(self):
        idx_l = self.env_cfg["joint_names"].index("left_joint_2")
        idx_r = self.env_cfg["joint_names"].index("right_joint_2")
        hip_l = self.dof_pos[:, idx_l]
        hip_r = self.dof_pos[:, idx_r]
        t = self.episode_length_buf.float() * self.dt
        w = 2 * math.pi / self.reward_cfg.get("period", 1.0)
        target = self.reward_cfg.get("bird_hip_target", -0.35)
        amp    = self.reward_cfg.get("bird_hip_amp", 0.15)
        sigma  = self.reward_cfg.get("bird_hip_sigma", 0.1)
        desired_l = target + amp * torch.sin(w * t)
        desired_r = target + amp * torch.sin(w * t + math.pi)
        err = 0.5 * ((hip_l - desired_l)**2 + (hip_r - desired_r)**2)
        return torch.exp(-err / (2 * sigma**2)).clamp(0.0, 1.0)
    

    # ============================================================
    # 11. Forward torso pitch (lean slightly forward)
    # ============================================================
    def _reward_forward_torso_pitch(self):
        pitch = self.base_euler[:, 1]
        target = self.reward_cfg.get("pitch_target", 0.17)
        sigma = self.reward_cfg.get("pitch_sigma", 0.1)
        err = (pitch - target)**2
        return torch.exp(-err / (2 * sigma**2))


    # ============================================================
    # 12. Hip abduction penalty (prevent sideways legs)
    # ============================================================
    def _reward_hip_abduction_penalty(self):
        try:
            idx_l = self.env_cfg["joint_names"].index("left_joint_1")
            idx_r = self.env_cfg["joint_names"].index("right_joint_1")

            abd_l = self.dof_pos[:, idx_l]
            abd_r = self.dof_pos[:, idx_r]

            err = abd_l**2 + abd_r**2
            sigma = self.reward_cfg.get("hip_abduction_sigma", 0.1)

            return torch.exp(-err / (2 * sigma**2))
        except:
            return torch.zeros((self.num_envs,), device=self.device)


    # ============================================================
    # 13. Lateral drift penalty (keep walking straight)
    # ============================================================
    def _reward_lateral_drift_penalty(self):
        drift = self.base_lin_vel[:, 1].abs()
        sigma = self.reward_cfg.get("drift_sigma", 0.1)
        return torch.exp(-drift**2 / (2 * sigma**2))


    # ============================================================
    # 14. Energy penalty (limit torque change)
    # ============================================================
    def _reward_energy_penalty(self):
        diff = torch.sum((self.actions - self.last_actions)**2, dim=1)
        sigma = self.reward_cfg.get("energy_sigma", 1.0)
        return torch.exp(-diff / (2 * sigma**2))


    # ============================================================
    # 15. Action rate penalty (smooth actions)
    # ============================================================
    def _reward_action_rate(self):
        return torch.sum((self.actions - self.last_actions)**2, dim=1)

