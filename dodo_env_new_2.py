import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Dodoenvironment:
    CONTACT_HEIGHT = 0.05
    SWING_HEIGHT_THRESHOLD = 0.10

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # basic config - sets up the fundamental "rules" and dimensions for your reinforcement learning (RL) training
        self.num_envs = num_envs    #instead of training one robot you are training multiple robots in parallel (environments)
        self.device = gs.device
        self.env_cfg = env_cfg  #saving the configuration dictionaries passed from the training script so the parameters can be accessed in class methods
        self.obs_cfg = obs_cfg
        self.num_privileged_obs = None
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        #These lines define the shape of the input and output layers for your Neural Network (policy):
        self.num_actions = env_cfg["num_actions"]   #The output size (motors). The policy outputs this many numbers (motor targets, torques, etc.) 
        self.num_obs = obs_cfg["num_obs"]   #The input size (sensors). The policy sees this many numbers (position, velocity, etc.)
        self.num_commands = command_cfg["num_commands"] #The goal input. Usually 3 for locomotion (Forward Velocity, Side Velocity, yaw Rate).

        self.dt = 0.02  #simulation timestep, in seconds
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True) #simulates delay that happens in real robots by using last timestep’s actions to mimic the delay

        self.obs_scales = obs_cfg["obs_scales"]     #neural network doesnt work well with large numbers, so observations/inputs are scaled to be roughly in range -1 to 1. 
        self.reward_scales = reward_cfg["reward_scales"]    #rewards scaled so that the learning algorithm is stable

        # create scene
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
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=64,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        try:
            self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        except Exception:
            self.scene.add_entity(gs.morphs.Plane(fixed=True))


        # add robot
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

        # build
        self.scene.build(n_envs=num_envs)

        ''' names to indices - bridge between neural network and the physics engine
        the disconnect is that we define the robot joints by name in the URDF file, but the physics engine (Genesis) refers to them by numbers in a flat array. 
        When the neural network outputs an action, it is a list of numbers (one per joint). we need to know which number corresponds to which joint. this mapping is done here.
        '''
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]


        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)    #applies the same proportional gain (kp) to all motors (num_actions), for the joints located at self.motors_dof_idx
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)    #applies the same derivative gain (kd) to all motors (num_actions), for the joints located at self.motors_dof_idx

        
        # ankle links for foot height measures
        self.ankle_links = []
        for ln in self.env_cfg.get("foot_link_names", []):
            try:
                self.ankle_links.append(self.robot.get_link(ln))
            except Exception:
                # ignore if link not found
                pass



        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()       #reward_functions is a pointer to the actual python function and episode_sums keeps track of the cumulative reward for each type
        for name in self.reward_scales.keys():
            # self.reward_scales[name] *= self.dt             # This turns “reward per second” → “reward per timestep”. we dont want to do this here, instead do it in the step() function when computing rewards
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        

        # initialize buffers - it is the short term memory of the simulation, storing current state information needed for control and learning
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)  #what your RL policy receives as input. It packs everything together: gravity, commands, joint positions, velocities, etc.
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)   #reward per environment buffer. The "score" for the current step. Calculated by summing up all those reward functions.
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)    #1 means "this env should reset on the next step". A list of flags (0 or 1). If reset_buf[5] == 1, it means Robot #5 fell over or ran out of time and needs to be reset to the starting position on the next step.
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)  #counts steps to enforce max episode length
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)    #the commands in command_cfg are stored here. stores what the robot should be trying to do (target velocities)
        self.commands_scale = torch.tensor( 
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],   #your policy samples commands. Commands go into the observation, and observations need normalization. lin_vel and ang_vel come from the obs_scales in obs_cfg         
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)  #actions output by the policy in the last timestep. The command the neural network just outputted (move leg to X).
        self.last_actions = torch.zeros_like(self.actions)  #used for smoothness penalty (action rate). The command from the previous step.
        self.dof_pos = torch.zeros_like(self.actions)   #current joint positions
        self.dof_vel = torch.zeros_like(self.actions)   #current joint velocities
        self.last_dof_vel = torch.zeros_like(self.actions)  #last timestep’s velocity (used for acceleration penalty etc.)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.current_ankle_heights = torch.zeros((self.num_envs, max(1, len(self.ankle_links))), device=gs.device, dtype=gs.tc_float)   #stores foot height for swing detection
        self.prev_contact = torch.zeros((self.num_envs, max(1, len(self.ankle_links))), device=gs.device, dtype=gs.tc_float)    # stores previous contact state for detecting touch-down events. is used for “contact consistency” and “early touchdown penalty”

        self.default_dof_pos = torch.tensor(    #at every environment reset, joints are set to these default angles. meaning your robot always restarts in a standing pose, not collapsed on the ground
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )


        '''
        In Reinforcement Learning (especially with libraries like rsl_rl or standard PPO implementations), the training loop expects the environment to return four things:
            Observations (What the robot sees)
            Rewards (The score)
            Resets (Game over flags)
            Extras (Everything else)


        Think of self.extras (dashboard) as a dictionary where you dump statistics that you want to graph later in WandB, but that the robot doesn't use to make decisions.

        The line self.extras["observations"] = dict() is setting up the infrastructure for the Critic.

        In PPO, there are two neural networks:
            The Actor (Policy): Decides what action to take. It sees self.obs_buf.
            The Critic (Value Function): Estimates how good that action was. It needs more information to make a better estimate.

        Sometimes, we want the Critic to see more (or different) information than the Actor (this is called Asymmetric Actor-Critic).
            Example: The Actor (Robot) only sees its sensors. The Critic (Teacher) sees the sensors plus the friction of the ground and the exact weight of the payload.
            By creating self.extras["observations"], you are preparing a container to send specific data to the Critic.
        '''
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()    #creating the critic's observation dictionary inside extras

        # initialize commands (random initial commands)
        self._resample_commands(torch.arange(self.num_envs, device=gs.device))


    # ---------------- command resampling ----------------
    def _resample_commands(self, envs_idx): # This function generates random target commands for each environment:
        if len(envs_idx) == 0:  #envs_idx is a list of envs that need new commands (usually when reset). If the list of environments to update is empty, do nothing.
            return

        ranges = self.command_cfg.get("command_ranges", {})
        lx = ranges.get("lin_vel_x", [0.0, 0.0])
        ly = ranges.get("lin_vel_y", [0.0, 0.0])
        ay = ranges.get("ang_vel_yaw", [0.0, 0.0])

        self.commands[envs_idx, 0] = gs_rand_float(lx[0], lx[1], (len(envs_idx),), self.device) #For these environments, update their lin_vel_x target command
        self.commands[envs_idx, 1] = gs_rand_float(ly[0], ly[1], (len(envs_idx),), self.device) #For these environments, update their lin_vel_y target command
        self.commands[envs_idx, 2] = gs_rand_float(ay[0], ay[1], (len(envs_idx),), self.device) #For these environments, update their ang_vel_yaw target command


    '''
    This block is the "Input -> Physics" phase of the simulation loop. It takes the decision from the brain (Neural Network) and turns it into physical movement in the world.

    Summary of this block: Brain output → Safety Clip → Add Delay → Convert to Angles → Send to Motors → Physics Happens.
    '''
    def step(self, actions):    # This function applies the actions, steps the simulation, computes observations, rewards, and checks for resets. One iteration of PPO
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]) #without this a random spike in actions could send the robot flying off into space
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions  #simulate action latency by using last actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos #Actions are offsets from the default pose, not absolute positions. If default hip angle = 0.4 rad, and the action gives +0.1 rad → the commanded pose is 0.5 rad. This is the standard PD offset formulation used in modern locomotion learning.
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)   #send commands to Genesis actuator API (position control). Sends the PD-controlled joint targets to Genesis. It tells the Genesis physics engine: "For the motors at self.motors_dof_idx, set the target position (P-gain target) to target_dof_pos."
        self.scene.step()


        # update buffers - after the physics step, we need to update our short term memory (buffers) with the latest state information from Genesis, so that the neural network can make informed decisions on the next step.
        self.episode_length_buf += 1    #increments the counter for how long the robot has been alive. we use this to force a reset or when to change the commands (resample)
        self.base_pos[:] = self.robot.get_pos() #get the per env robot base position from Genesis. in the world frame
        self.base_quat[:] = self.robot.get_quat() #get the per env robot base orientation from Genesis. in the world frame (quaternion)
        self.base_euler = quat_to_xyz(      #Calculating tilt. Transform orientation into the robot’s initial frame. Convert to roll, pitch, yaw in degrees. Used for termination: too much pitch → episode ends.
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)   #get linear velocity in robot local frame. forward velocity is base_lin_vel[0](x axis), base_lin_vel[1](y axis) lateral, base_lin_vel[2](z axis) yaw. transform_by_quat rotates the World Velocity vector into the Robot's Body Frame.
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)  #If the robot tips forward/backward, gravity projected into its frame changes accordingly. This helps PPO learn balance.

        #Proprioception: The sense of where your limbs are. These are passed directly to the neural network so it knows the configuration of its legs.
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)     #Angles of every motor (e.g., Hip is at 0.4 rad). Used in tracking reward & observations.
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)     #Speed of every motor (e.g., Knee is extending at 2.0 rad/s). 
        
        # ankle heights
        if len(self.ankle_links) > 0:
            try:
                self.current_ankle_heights[:] = torch.stack([link.get_pos()[:, 2] for link in self.ankle_links], dim=1) #Gets z-height of left/right ankles
            except Exception:
                # if link.get_pos isn't batched, call per-env and stack -- skip to avoid slowdown
                pass
        
        '''
        resample commands - changing the mission. We want it to handle transitions. For example: "Walk Fast" → "Stop" → "Turn Left". Handle different commands during one episode.

        resampling_time_s: How often the command changes (e.g., every 4.0 seconds).

        .nonzero(as_tuple=False)
        .reshape((-1,))
        this bit extracts the indices of the environments that need new commands based on the episode length and the resampling time.
        '''
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)  
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)


        '''
        This section defines the "Game Over" rules. In Reinforcement Learning, we need to know when to stop the simulation and start over, either because the robot succeeded (time's up) or failed (fell over).
        self.reset_buf is a list of flags (0 or 1). If reset_buf[5] == 1, it means Robot #5 fell over or ran out of time and needs to be reset to the starting position on the next step.
        '''
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length  #If the robot has survived for the maximum allowed time, mark it for reset.
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]  #If roll or pitch exceed threshold → robot fell.
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        

        '''
        handles the bookeeping for resets
        It is critical to distinguish why a robot is resetting. Did it fail (fall over)? Or did it succeed (survive until the time limit)? The RL algorithm handles these two cases very differently.
        '''
        # mark timeouts for extras
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))   # get indices of envs that timed out (not fell). Used by PPO to mask terminal states.
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))   #Reset the environments that need restarting. Extracs the indices of the environments to reset (which is part of self.reset_buf) and calls reset_idx()


        '''
        calculates 'grade' for the robot's performance in this exact step. In RL this is the only way the robot knows if it is doing well or poorly. 

        reward_func() is the raw score and calls the specific reward function (e.g., _reward_tracking_lin_vel). It returns the raw score usually based on physics errors (e.g., velocity error).

        self.reward_scales[name] asks how much does this reward matter? Track velocity is more important than energy efficiency, so it gets a higher weight.

        self.dt is the time normalization factor. Rewards are often defined as "reward per second", but since our simulation runs in discrete time steps, we multiply by dt to convert it to "reward per timestep".

        self.rew_buf += rew adds up all the individual reward components into a single scalar reward for the current step for the PPO neural network to use.

        self.episode_sums[name] += rew is for the human to track how much total reward of each type the robot has accumulated over the entire episode. Useful for logging and debugging.

        '''
        # compute reward
        self.rew_buf[:] = 0.0   #we 0 out the buffer at the start of each step so we don't accidentally add points from the past. 
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name] * self.dt    #multiplying by dt to get “reward per second” → “reward per timestep”
            self.rew_buf += rew
            self.episode_sums[name] += rew

                        
        '''
        computes observations - this is what the neural network sees as input to make its next decision. It packs everything together: gravity, commands, joint positions, velocities, etc.
        The Neural Network (the Policy) is mathematically just a big function that takes one single list of numbers as input. It cannot accept a dictionary or separate variables.
        Seperate columns for seperate features, all stacked side by side into one big row vector.
        axis=-1: This ensures we stack the features (columns), not the robots (rows).
        '''
        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3, added this 27.11, forgot to get lin vel into obs
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 8   #We dont feed the absolute joint angles to the neural network. Instead we feed the difference from the default standing pose. t makes learning easier. Zero means "Perfect Standing Pose." The network learns deviations from that stable shape.
                self.dof_vel * self.obs_scales["dof_vel"],  # 8
                self.actions,  # 8
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:] #These lines save the current state into a "Previous State" buffer before the function ends. This data is critical for the next timestep (t+1).


        '''
        Feeding the teacher (the critic)

        The Actor (Student): Uses self.obs_buf to decide what to do.
        The Critic (Teacher): Uses this entry to judge how good that decision was.
        Currently: You are passing the exact same info (obs_buf) to the Critic.
        Future Potential: If you later want to give the Critic "God Mode" (e.g., exact friction values, terrain heightmap), you would create a different buffer (e.g., privileged_obs_buf) and assign it here instead. This line keeps your code flexible.
        '''
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):  # This function resets specific environments (robots) to their initial state. Called when a robot falls or the episode ends.
        if len(envs_idx) == 0:      #in a single step if no envs need reset, stop immediately
            return
        
        # reset dofs (the limbs/joints)
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)


    '''
    this is the global hard reset function that resets all environments at once. It is called at the very start of training only once throughout to ensure all robots begin from a known state.
    '''
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None



    #_______________REWARD FUNCTIONS_______________#
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
        thr_r = self.reward_cfg.get("roll_threshold", 30 * math.pi / 180)
        thr_p = self.reward_cfg.get("pitch_threshold", 30 * math.pi / 180)
        upright = ((roll < thr_r) & (pitch < thr_p)).float()
        return upright

    # ============================================================
    # 6. Fall penalty (roll/pitch exceeding threshold)
    # ============================================================
    def _reward_fall_penalty(self):
        roll = self.base_euler[:, 0].abs()
        pitch = self.base_euler[:, 1].abs()

        thr_r = self.reward_cfg.get("roll_threshold", 30 * math.pi / 180)
        thr_p = self.reward_cfg.get("pitch_threshold", 30 * math.pi / 180)

        fail = ((roll > thr_r) | (pitch > thr_p)).float()
        return -fail


    # ============================================================
    # 7. Periodic gait reward (left stance → right stance)
    # ============================================================
    def _reward_periodic_gait(self):
        period = self.reward_cfg.get("period", 1.0)
        phase = (self.episode_length_buf.float() * self.dt) % period
        half = period * 0.5
        contact = torch.zeros((self.num_envs, max(1, len(self.ankle_links))), device=self.device)
        if self.current_ankle_heights.numel():
            contact = (self.current_ankle_heights < self.CONTACT_HEIGHT).float()
        desired_left = (phase < half).float()
        desired_right = (phase >= half).float()
        if contact.shape[1] == 1:
            return (desired_left * contact[:, 0] + desired_right * contact[:, 0]).clamp(0.0, 1.0)
        return (desired_left * contact[:, 0] + desired_right * contact[:, 1]).clamp(0.0, 1.0)


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
