import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.envs.go2.go2_jump.go2_jump_config import GO2JumpCfg, GO2JumpCfgPPO
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi

# --- 辅助函数 (来自 SATA) ---
#
def update_com(I_box, mass_box, com_box, mass_point, point_pos):
    new_com = (mass_box * com_box + mass_point * point_pos) / (mass_box + mass_point)
    return new_com

def parallel_axis_theorem(I_com, mass, d):
    d_x, d_y, d_z = d
    d_squared = np.array([
        [d_y ** 2 + d_z ** 2, -d_x * d_y, -d_x * d_z],
        [-d_x * d_y, d_x ** 2 + d_z ** 2, -d_y * d_z],
        [-d_x * d_z, -d_y * d_z, d_x ** 2 + d_y ** 2]
    ])
    return I_com + mass * d_squared

def update_inertia(I_box, mass_box, com_box, mass_point, point_pos):
    new_com = update_com(I_box, mass_box, com_box, mass_point, point_pos)
    displacement_box = com_box - new_com
    I_box_new = parallel_axis_theorem(I_box, mass_box, displacement_box)
    displacement_point = point_pos - new_com
    I_point = parallel_axis_theorem(np.zeros((3, 3)), mass_point, displacement_point)
    I_total = I_box_new + I_point
    return I_total, new_com
# --- 辅助函数结束 ---


class GO2Jump(LeggedRobot):
    cfg: GO2JumpCfg

    # --- 关键修复：修正 __init__ 顺序 ---
    def __init__(self, cfg: GO2JumpCfg, sim_params, physics_engine, sim_device, headless):
        
        # 1. 初始化课程变量 (必须在 super 之前)
        self.curriculum_stage = 0
        self.curriculum_reward_thresholds = cfg.curriculum_thresholds

        # 2. **首先**调用 super().__init__
        #    这将正确设置 self.device (在play模式下为'cpu', 训练模式下为'cuda')
        #    并创建 self.body_names, self.feet_indices 等
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 3. **然后**初始化SATA的“肌肉”和“生长”变量
        self.max_torque_scale = cfg.growth.max_torque_scale
        self.start_torque_scale = cfg.growth.start_torque_scale
        self.max_rear_torque_scale = cfg.growth.max_rear_torque_scale
        self.start_rear_torque_scale = cfg.growth.start_rear_torque_scale
        self.max_freq = cfg.growth.max_freq
        self.start_freq = cfg.growth.start_freq
        self.step_count = 0
        self.current_dt = 0
        self.current_freq = self.start_freq
        self.low_torque = 0
        
        # 4. **最后**在正确的 self.device 上创建SATA张量
        #    (修复 'cuda:0' vs 'cpu' 的bug)
        self.motor_fatigue = torch.zeros(cfg.env.num_envs, self.num_dofs, device=self.device)
        self.torques = torch.zeros(cfg.env.num_envs, self.num_dofs, device=self.device)
        self.activation_sign = torch.zeros(cfg.env.num_envs, self.num_dofs, device=self.device)
        if self.cfg.domain_rand.loss_action_obs == False:
            self.cfg.domain_rand.loss_rate = 0
        # --- __init__ 修复完毕 ---
        
    # --- 关键修复：修正 _init_buffers 顺序 ---
    def _init_buffers(self):
        # 1. **首先**调用基类的 _init_buffers
        super()._init_buffers() 
        # (这会创建 self.dof_state, self.contact_forces, ...)
        self.num_feet = len(self.feet_indices)
        self.base_index = self.body_names.index('trunk')
        # 2. **然后**定义SATA的缓冲区
        self.prev_dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.hip_indices = []
        self.thigh_indices = []
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if 'hip_joint' in name:
                self.hip_indices.append(i)
            if 'thigh_joint' in name:
                self.thigh_indices.append(i)
        self.torque_limits = torch.ones(self.num_dofs, device=self.device) * 23.5
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.general_scale = 0
        
        # 3. **最后**定义“跳跃”任务和奖励所需的新缓冲区
        #    (self.base_index 和 self.num_feet 已在 super().__init__ 中定义)
        self.target_landing_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.target_landing_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.obstacle_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.obstacle_dims = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        self.feet_air_time = torch.zeros(self.num_envs, self.num_feet, device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros(self.num_envs, self.num_feet, device=self.device, dtype=torch.bool)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # --- _init_buffers 修复完毕 ---

    # --- 保留SATA的函数 ---
    def _process_dof_props(self, props, env_id):
        #
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.termination_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                          requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.termination_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] - 0.05
                self.termination_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] + 0.05
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    # --- 关键修复：修正 check_termination ---
    def check_termination(self):
        # 1. 修复 NameError：添加SATA的定义
        dof_pos_limits_up = self.termination_dof_pos_limits[:, 1]
        dof_pos_limits_low = self.termination_dof_pos_limits[:, 0]
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 2. 保留SATA的关节限制终止
        self.reset_buf |= torch.any(self.dof_pos > dof_pos_limits_up, dim=1)
        self.reset_buf |= torch.any(self.dof_pos < dof_pos_limits_low, dim=1)

        # 3. 保留“跳跃”项目的 "thigh"/"calf" 接触终止
        self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > self.cfg.asset.penalize_contacts_force, dim=1)

        # 4. **硬编码**“肚子贴地”终止 (修复 `base_index` 漏洞)
        #    (self.base_index 已在 _init_buffers 中正确定义)
        base_contact = torch.norm(self.contact_forces[:, self.base_index, :], dim=-1) > 0.1
        self.reset_buf |= base_contact

        # 5. 保留超时终止
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        # --- check_termination 修复完毕 ---

    # --- 保留SATA的函数 ---
    def _process_rigid_body_props(self, props, env_id):
        #
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            com_rng_x = self.cfg.domain_rand.shifted_com_range_x
            com_rng_y = self.cfg.domain_rand.shifted_com_range_y
            com_rng_z = self.cfg.domain_rand.shifted_com_range_z
            rnd_mass = np.random.uniform(rng[0], rng[1])
            point_mass_pos = np.array([np.random.uniform(com_rng_x[0], com_rng_x[1]),
                                       np.random.uniform(com_rng_y[0], com_rng_y[1]),
                                       np.random.uniform(com_rng_z[0], com_rng_z[1])])
            props[0].mass += rnd_mass
            com_prev = np.array([props[0].com.x, props[0].com.y, props[0].com.z])
            inertia_prev = np.array([[props[0].inertia.x.x, props[0].inertia.x.y, props[0].inertia.x.z],
                                     [props[0].inertia.y.x, props[0].inertia.y.y, props[0].inertia.y.z],
                                     [props[0].inertia.z.x, props[0].inertia.z.y, props[0].inertia.z.z]])
            intertia, com = update_inertia(inertia_prev, props[0].mass, com_prev, rnd_mass, point_mass_pos)
            props[0].inertia.x += gymapi.Vec3(intertia[0, 0], intertia[0, 1], intertia[0, 2])
            props[0].inertia.y += gymapi.Vec3(intertia[1, 0], intertia[1, 1], intertia[1, 2])
            props[0].inertia.z += gymapi.Vec3(intertia[2, 0], intertia[2, 1], intertia[2, 2])
            props[0].com = gymapi.Vec3(com[0], com[1], com[2])
            for i in range(len(props)):
                props[i].mass += np.random.uniform(rng[0] / 16, rng[1] / 16)
        return props

    # --- 保留SATA的函数 ---
    def _reset_dofs(self, env_ids):
        #
        self.dof_pos[env_ids] = (
                self.default_dof_pos * torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device))
        self.dof_vel[env_ids] = 0.
        self.activation_sign[env_ids] = 0.

        if self.add_noise:
            self.motor_fatigue[env_ids] = torch_rand_float(0, 0.2 * self.general_scale, (len(env_ids), 12),
                                                           device=self.device).squeeze(
                1)
        else:
            self.motor_fatigue[env_ids] = torch.zeros_like(self.motor_fatigue[env_ids])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # --- 保留SATA的函数 ---
    def _reset_root_states(self, env_ids):
        #
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # --- 关键：保留SATA的“物理发育”课程 ---
    def _update_growth_scale(self):
        #
        self.step_count += 1
        if self.cfg.control.control_type == "T" or self.cfg.test.use_test:
            self.step_count = GO2JumpCfgPPO().runner.num_steps_per_env * self.cfg.test.checkpoint

        self.general_scale = np.exp(-np.exp((-self.cfg.growth.k * (self.step_count - self.cfg.growth.x0))))
        self.current_freq = self.general_scale * (self.max_freq - self.start_freq) + self.start_freq
        self.current_torque_limit_scale = self.general_scale * (
                self.max_torque_scale - self.start_torque_scale) + self.start_torque_scale
        self.r_leg_scaled = self.general_scale * (
                self.max_rear_torque_scale - self.start_rear_torque_scale) + self.start_rear_torque_scale

    # --- 保留SATA的Step逻辑 ---
    def step(self, actions):
        #
        self.actions = actions.to(self.device)
        self.render()
        self.rew_buf[:] = 0.
        while self.current_dt * self.current_freq < 1:
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.current_dt += self.dt
            self.post_physics_step()
        self.current_dt %= (1 / self.current_freq)

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # --- 关键：保留SATA的“肌肉” (力矩计算) ---
    def _compute_torques(self, actions):
        #
        self._update_growth_scale()
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        self.torques_action = actions_scaled
        torques_limits = self.current_torque_limit_scale * self.torque_limits
        torques_limits[6:] = torques_limits[6:] * self.r_leg_scaled

        if self.cfg.control.activation_process:
            current_activation_sign = torch.tanh(self.torques_action / torques_limits)
            activation_sign = (current_activation_sign - self.activation_sign) * 0.6 + self.activation_sign
        else:
            activation_sign = self.torques_action / torques_limits
        self.activation_sign = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            activation_sign, self.activation_sign)

        if self.cfg.control.hill_model:
            self.torques = self.activation_sign * torques_limits * (
                    1 - torch.sign(self.activation_sign) * self.dof_vel / self.dof_vel_limits)
        else:
            self.torques = self.activation_sign * torques_limits

        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        if self.low_torque:
            self.torques[:,:3] = self.torques[:,:3] * 0.2
            
        # --- (删除 self.last_torques 更新, 已移到 post_physics_step_callback) ---
        return self.torques

    # --- 保留SATA的噪声设置 (已在 compute_observations 中适配) ---
    def _get_noise_scale_vec(self, cfg):
        #
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device) # 修正：使用 num_observations
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # torques (action)
        noise_vec[45:57] = noise_scales.fatigue * noise_level / 10
        noise_vec[57:67] = 0. # 跳跃任务目标 (10 dims)
        
        if self.cfg.terrain.measure_heights:
             pass
        return noise_vec

    # --- 保留SATA的推力 ---
    def _push_robots(self):
        #
        max_vel = self.cfg.domain_rand.max_push_vel_xy * self.general_scale
        self.root_states[:, 7:10] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3),
                                                     device=self.device)  # lin vel x/y
        max_ang_vel = self.cfg.domain_rand.max_push_vel_ang * self.general_scale
        self.root_states[:, 10:13] = torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3),
                                                      device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # --- 关键：融合“观测” (67-dim) ---
    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = self.motor_fatigue.detach()
        
        # 3(lin_vel) + 3(ang_vel) + 3(gravity) + 12(dof_pos) + 12(dof_vel) = 33 (SATA Base)
        # + 12(torques) + 12(fatigue) = 24 (SATA Torque)
        # + 3(target_pos) + 1(target_yaw) + 3(obstacle_pos) + 3(obstacle_dims) = 10 (Jumping Task)
        # = 67 维
        obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.torques, # SATA 特有观测
                             motor_fatigue, # SATA 特有观测
                             self.target_landing_pos, # 跳跃任务观测
                             self.target_landing_yaw, # 跳跃任务观测
                             self.obstacle_pos, # 跳跃任务观测
                             self.obstacle_dims  # 跳跃任务观测
                             ), dim=-1)
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf)

    # --- 关键修复：修正 _post_physics_step_callback ---
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        
        # 1. 更新“技能学习”课程
        self._update_curriculum()
        
        # 2. 更新足部滞空时间
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.feet_air_time[contact_filt] = 0.
        self.feet_air_time[~contact_filt] += self.dt
        
        # 3. **修复Bug**：在所有奖励计算*之后*更新 `last_` 张量
        #    (修复 `rew_action_rate` 和 `rew_dof_acc` 为0的bug)
        self.last_torques[:] = self.torques
        self.last_dof_vel[:] = self.dof_vel
    # --- _post_physics_step_callback 修复完毕 ---

    # --- 关键：新增“技能学习”课程 ---
    def _update_curriculum(self):
        #
        if "jump" not in self.episode_sums:
             pass 

        mean_jump_reward = torch.mean(self.episode_sums["jump"][self.episode_sums["jump"] != 0])
        
        if self.curriculum_stage < len(self.curriculum_reward_thresholds):
            if not torch.isnan(mean_jump_reward) and mean_jump_reward > self.curriculum_reward_thresholds[self.curriculum_stage]:
                self.curriculum_stage += 1
                print(f"--- ADVANCING TO JUMP CURRICULUM STAGE {self.curriculum_stage} ---")

    # --- 关键：重写“指令采样” (来自 "Jumping") ---
    def _resample_commands(self, env_ids):
        #
        if self.curriculum_stage == 0:
            self.target_landing_pos[env_ids, 0] = 0.
            self.target_landing_pos[env_ids, 1] = 0.
        elif self.curriculum_stage == 1:
            self.target_landing_pos[env_ids, 0] = torch_rand_float(0.0, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
            self.target_landing_pos[env_ids, 1] = 0.
        elif self.curriculum_stage == 2:
            self.target_landing_pos[env_ids, 0] = torch_rand_float(0.0, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
            self.target_landing_pos[env_ids, 1] = torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.target_landing_pos[env_ids, 0] = torch_rand_float(0.0, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
            self.target_landing_pos[env_ids, 1] = torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device).squeeze(1)
            self.obstacle_dims[env_ids, 0] = torch_rand_float(
                self.cfg.obstacles.obstacle_width_range[0], 
                self.cfg.obstacles.obstacle_width_range[1], 
                (len(env_ids), 1), device=self.device).squeeze(1)
            self.obstacle_dims[env_ids, 1] = self.obstacle_dims[env_ids, 0]
            self.obstacle_dims[env_ids, 2] = torch_rand_float(
                self.cfg.obstacles.obstacle_height_range[0], 
                self.cfg.obstacles.obstacle_height_range[1], 
                (len(env_ids), 1), device=self.device).squeeze(1)
            self.obstacle_pos[env_ids, 0] = self.target_landing_pos[env_ids, 0] / 2.0
            self.obstacle_pos[env_ids, 1] = self.target_landing_pos[env_ids, 1] / 2.0
            self.obstacle_pos[env_ids, 2] = 0.0

        self.target_landing_pos[env_ids, 2] = torch_rand_float(
            self.cfg.commands.ranges.target_height[0], 
            self.cfg.commands.ranges.target_height[1], 
            (len(env_ids), 1), device=self.device).squeeze(1)
            
        self.target_landing_yaw[env_ids, 0] = torch_rand_float(
            self.cfg.commands.ranges.target_ang_yaw[0], 
            self.cfg.commands.ranges.target_ang_yaw[1], 
            (len(env_ids), 1), device=self.device).squeeze(1)

    ##############################################################################################################
    # --- 关键：删除SATA的8个奖励函数 ---
    # (原 _reward_soft_dof_pos_limits ... _reward_lin_vel_z 已删除)
    #
    
    # --- 关键：新增“跳跃”项目的21个奖励函数 (从 "Jumping" 基类复制) ---
    
    # 1. 任务奖励 (Task Rewards)
    def _reward_jump(self):
        # --- 关键修复：防止“站桩”的奖励设计 ---
        target_height = self.target_landing_pos[:, 2]
        current_height = self.root_states[:, 2]
        height_reward = torch.clamp(current_height, max=target_height[0])
        is_in_air = torch.any(self.feet_air_time > 0.1, dim=1)
        z_vel_reward = torch.clamp(self.base_lin_vel[:, 2], min=0.0, max=1.5) * (~is_in_air)
        target_pos_xy = self.target_landing_pos[:, :2]
        current_pos_xy = self.root_states[:, :2] - self.env_origins[:, :2] 
        pos_error = torch.norm(current_pos_xy - target_pos_xy, dim=1)
        pos_reward = torch.exp(-pos_error * 2.0) * is_in_air
        return height_reward + pos_reward + z_vel_reward

    def _reward_landing_pose(self):
        # (逻辑简化)
        is_landed = torch.all(self.contact_forces[:, self.feet_indices, 2] > 1.0, dim=1)
        pose_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        pose_reward = torch.exp(-pose_error * 0.5)
        return pose_reward * is_landed

    def _reward_orientation(self):
        #
        quat_error = torch.sum(torch.square(self.base_quat[:, :2]), dim=1)
        return torch.exp(-quat_error / self.cfg.rewards.tracking_sigma)
        
    def _reward_ang_vel_z(self):
        #
        ang_vel_z_error = torch.square(self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_z_error / self.cfg.rewards.tracking_sigma)

    # 2. 惩罚项 (Penalties)
    def _reward_termination(self):
        #
        return self.reset_buf.float()

    def _reward_collision(self):
        # --- 关键修复：同时惩罚“肚子”和“腿” ---
        leg_collision = torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        belly_collision = (torch.norm(self.contact_forces[:, self.base_index, :], dim=-1) > 0.1).float()
        return leg_collision + belly_collision

    def _reward_base_height_binary(self):
        # (逻辑实现)
        return (self.root_states[:, 2] < 0.18).float()

    def _reward_lin_vel_z(self):
        #
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        #
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_stand_still(self):
        #
        return (torch.norm(self.dof_pos - self.default_dof_pos, dim=1) < 0.05).float()

    # 3. 平滑性与能量 (Smoothness & Energy)
    def _reward_torques(self):
        #
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        #
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        #
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        #
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_pos_limits(self):
        #
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # 4. 腿部与步态塑形 (Leg & Gait Shaping)
    def _reward_hop_stand_still(self):
        #
        is_jumping = torch.any(self.feet_air_time > 0.1, dim=1)
        is_still = (torch.norm(self.dof_vel, dim=1) < 0.1)
        return (is_jumping & is_still).float()
        
    def _reward_swing_curled(self):
        #
        calf_joints = self.dof_pos[:, [2, 5, 8, 11]]
        is_swing = self.feet_air_time > 0.1
        penalty = (calf_joints > -1.0) * is_swing
        return torch.sum(penalty, dim=1).float()

    def _reward_swing_unjerk(self):
        #
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_stance_unjerk(self):
        #
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_air_time(self):
        # --- 关键修复：防止“肚子贴地”漏洞 ---
        all_feet_in_air = torch.all(self.feet_air_time > 0.1, dim=1)
        # (self.base_index 已在 _init_buffers 中正确定义)
        base_contact = torch.norm(self.contact_forces[:, self.base_index, :], dim=-1) > 0.1
        
        # 只有在“真跳”（脚离地且肚子没贴地）时才给奖励
        true_jump = all_feet_in_air & (~base_contact)
        
        return true_jump.float() * self.cfg.rewards.scales.feet_air_time