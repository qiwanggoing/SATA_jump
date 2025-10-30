from legged_gym.envs import LeggedRobot
from legged_gym.envs.go2.go2_jump.go2_jump_config import GO2JumpCfg, GO2JumpCfgPPO
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np 

# 继承SATA的 GO2Torque
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque

# (来自 go2_torque.py)
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
# (以上函数来自 go2_torque.py)


class GO2Jump(GO2Torque):
    cfg: GO2JumpCfg # 关联新的Config

    def _init_buffers(self):
        super()._init_buffers() 

        # (OmniNet灵感)
        self.aerial_tuck_angles = torch.tensor([self.cfg.rewards.aerial_tuck_angles[name] for name in self.dof_names], device=self.device)

        # 跳跃状态追踪
        self.in_air = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.air_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # (Atanassov灵感) 落地奖励缓冲区
        self.landing_dist_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([1.0, 1.0], device=self.device, requires_grad=False) 

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids) 
        self.air_time_buf[env_ids] = 0.0
        self.in_air[env_ids] = False
        self.landing_dist_rew[env_ids] = 0.0

    def _post_physics_step_callback(self):
        # (修复：打破SATA的 super() 继承链，避免行走逻辑的IndexError)

        # 1. (来自 LeggedRobot) 检查是否需要 pushing
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
            
        # 2. (来自 GO2Torque) 处理测试指令
        if self.cfg.test.use_test and self.cfg.control.control_type == "T":
            self.commands[:] = self.cfg.test.vel[:2] # 使用2D指令
            
        # 3. (我们自己的 GO2Jump 逻辑)
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] > 0.1, dim=1) > 0
        just_landed = (self.in_air) & (contacts)
        
        self.in_air = ~contacts
        self.air_time_buf += self.in_air * self.dt
        
        # (Atanassov灵感)
        self.landing_dist_rew[:] = 0. 
        if torch.any(just_landed):
            landing_dist_error = torch.square(self.root_states[just_landed, 0] - self.commands[just_landed, 1])
            self.landing_dist_rew[just_landed] = landing_dist_error * self.cfg.rewards.scales.landing_dist
        
        self.air_time_buf[just_landed] = 0.0

    # (Atanassov灵感)
    def _resample_commands(self, env_ids):
        """ 使用SATA的general_scale 来实现两阶段跳跃课程 """
        
        # 阶段 1: 调度目标高度
        min_height = self.command_ranges["target_height"][0]
        max_height = self.command_ranges["target_height"][1]
        scaled_max_height = min_height + (max_height - min_height) * self.general_scale
        self.commands[env_ids, 0] = torch_rand_float(min_height, scaled_max_height, (len(env_ids), 1), device=self.device).squeeze(1)

        # 阶段 2: 调度前向距离
        min_dist = self.command_ranges["target_forward_dist"][0]
        max_dist = self.command_ranges["target_forward_dist"][1]
        
        if self.general_scale > self.cfg.growth.forward_jump_threshold:
            forward_scale = (self.general_scale - self.cfg.growth.forward_jump_threshold) / (1.0 - self.cfg.growth.forward_jump_threshold)
            scaled_max_dist = min_dist + (max_dist - min_dist) * forward_scale
            self.commands[env_ids, 1] = torch_rand_float(min_dist, scaled_max_dist, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 1] = min_dist

    # (SATA)
    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = self.motor_fatigue.detach()
        
        obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.commands[:, :2] * self.commands_scale, # 2D 指令
                             self.torques,
                             motor_fatigue
                             ), dim=-1)
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec[:self.cfg.env.num_observations]

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf)

    # (SATA)
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:35] = 0.  # 2个指令
        noise_vec[35:47] = 0.  # torques
        noise_vec[47:59] = noise_scales.fatigue * noise_level / 10 # fatigue
        
        return noise_vec

    ##############################################################################################################
    # --- 新的跳跃奖励函数 (借鉴 Bilibili-go2) ---

    def _reward_stand(self):
        # (新) 奖励机器人回到蹲伏的默认姿态 (为下一次跳跃做准备)
        return torch.exp(-torch.norm(self.default_dof_pos - self.dof_pos, dim=-1))

    def _reward_jump_z_vel(self):
        # (新) 只要Z轴速度>0，就给予奖励
        return (self.base_lin_vel[:, 2]).clip(min=0.)

    def _reward_air_time(self):
        # (新) 只要在空中 (self.in_air 是 True)，就给予奖励
        return self.in_air.float()
        
    # --- 惩罚项 (在训练后期启动) ---

    def _reward_stability(self):
        # 惩罚 roll 和 pitch (仅在训练后期)
        penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return penalty * self.general_scale # (SATA 生长惩罚)
        
    def _reward_tracking_ang_vel(self):
        # 惩罚 roll 和 pitch 角速度 (仅在训练后期)
        penalty = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return penalty * self.general_scale # (SATA 生长惩罚)

    def _reward_landing_dist(self):
        # (Atanassov灵感) 惩罚落地误差 (仅在训练后期)
        return self.landing_dist_rew * self.general_scale # (SATA 生长惩罚)

    # (OmniNet灵感)
    def _reward_aerial_posture(self):
        # 惩罚空中姿态 (仅在训练后期)
        pose_error = torch.sum(torch.square(self.dof_pos - self.aerial_tuck_angles), dim=1)
        return pose_error * self.in_air.float() * self.general_scale # (SATA 生长惩罚)

    # --- 保留SATA的约束奖励 ---
    # (这些函数继承自 GO2Torque 和 LeggedRobot)
    # _reward_soft_dof_pos_limits
    # _reward_motor_fatigue
    # _reward_dof_acc