from legged_gym.envs import LeggedRobot
# 导入新的配置文件
from legged_gym.envs.go2.go2_jump.go2_jump_config import GO2JumpCfg, GO2JumpCfgPPO
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np # 确保导入 numpy

# 继承SATA的 GO2Torque
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque

# 确保 update_com 和 parallel_axis_theorem 函数存在 (从 go2_torque.py 复制)
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
# (以上函数来自 go2_torque.py)


# 修改类名
class GO2Jump(GO2Torque):
    cfg: GO2JumpCfg # 关联新的Config

    def _init_buffers(self):
        super()._init_buffers() # 调用父类 (GO2Torque) 的 _init_buffers

        # (OmniNet灵感)
        # 缓存"收腿"姿态
        self.aerial_tuck_angles = torch.tensor([self.cfg.rewards.aerial_tuck_angles[name] for name in self.dof_names], device=self.device)

        # 跳跃状态追踪
        self.in_air = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.air_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # (Atanassov灵感) 落地奖励缓冲区
        self.landing_dist_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 将指令维度修改为2 (target_height, target_forward_dist)
        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([1.0, 1.0], device=self.device, requires_grad=False) # 跳跃指令不需要SATA的速度缩放
        
        # 确保SATA的这些缓冲区仍然存在 (父类会创建它们)
        # self.motor_fatigue
        # self.torques
        # self.activation_sign

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids) # 调用父类 (SATA) 的重置
        # 重置跳跃追踪缓冲区
        self.air_time_buf[env_ids] = 0.0
        self.in_air[env_ids] = False
        self.landing_dist_rew[env_ids] = 0.0

    def _post_physics_step_callback(self):
        # 继承SATA的_post_physics_step_callback (用于域随机化等)
        #
        super()._post_physics_step_callback() 

        # 更新跳跃状态
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] > 0.1, dim=1) > 0
        just_landed = (self.in_air) & (contacts)
        
        self.in_air = ~contacts
        self.air_time_buf += self.in_air * self.dt
        
        # (Atanassov灵感)
        # 在刚落地时计算落地奖励
        self.landing_dist_rew[:] = 0. # 默认无奖励
        if torch.any(just_landed):
            # 计算X轴（前向）落地误差
            landing_dist_error = torch.square(self.root_states[just_landed, 0] - self.commands[just_landed, 1])
            # 奖励 = 误差 * 负的scale
            self.landing_dist_rew[just_landed] = landing_dist_error * self.cfg.rewards.scales.landing_dist
        
        # 刚落地时，重置空中时间
        self.air_time_buf[just_landed] = 0.0

    # (Atanassov灵感)
    # 彻底重写指令采样函数
    def _resample_commands(self, env_ids):
        """ 使用SATA的general_scale 来实现两阶段跳跃课程 """
        
        # --- 阶段 1: 学习原地垂直跳跃 ---
        # 调度目标高度
        min_height = self.command_ranges["target_height"][0]
        max_height = self.command_ranges["target_height"][1]
        # 使用 general_scale 缩放最大高度
        scaled_max_height = min_height + (max_height - min_height) * self.general_scale
        self.commands[env_ids, 0] = torch_rand_float(min_height, scaled_max_height, (len(env_ids), 1), device=self.device).squeeze(1)

        # --- 阶段 2: 学习向前跳跃 ---
        min_dist = self.command_ranges["target_forward_dist"][0]
        max_dist = self.command_ranges["target_forward_dist"][1]
        
        # 仅当 general_scale 超过阈值时，才开始调度前向距离
        if self.general_scale > self.cfg.growth.forward_jump_threshold:
            # 将 [threshold, 1.0] 映射到 [0.0, 1.0]
            forward_scale = (self.general_scale - self.cfg.growth.forward_jump_threshold) / (1.0 - self.cfg.growth.forward_jump_threshold)
            scaled_max_dist = min_dist + (max_dist - min_dist) * forward_scale
            self.commands[env_ids, 1] = torch_rand_float(min_dist, scaled_max_dist, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # 在此之前，只进行原地跳跃
            self.commands[env_ids, 1] = min_dist

    # 重写SATA的观测函数
    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = self.motor_fatigue.detach()
        
        # SATA的观测
        obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.commands[:, :2] * self.commands_scale, # 修改：使用2个新指令
                             self.torques,
                             motor_fatigue
                             ), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            # (注意: self.noise_scale_vec 需要在 _get_noise_scale_vec 中调整维度)
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec[:self.cfg.env.num_observations]

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf)

    # 调整SATA的 _get_noise_scale_vec 
    def _get_noise_scale_vec(self, cfg):
        # 继承SATA的噪声配置，但调整维度
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
    # --- 新的跳跃奖励函数 ---
    # (删除SATA的所有行走奖励)

    def _reward_stability(self):
        # 惩罚 roll 和 pitch
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
    def _reward_tracking_ang_vel(self):
        # 惩罚 roll 和 pitch 角速度
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_jump_z_vel(self):
        # 奖励Z轴起跳速度 (仅在起跳瞬间奖励)
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] > 0.1, dim=1) > 0
        just_took_off = (~self.in_air) & (~contacts)
        
        # 目标Z速度与目标高度相关 (h = v^2 / 2g) => v = sqrt(2 * g * h_cmd)
        target_z_vel = torch.sqrt(2 * 9.81 * self.commands[:, 0])
        vel_error = torch.square(self.base_lin_vel[:, 2] - target_z_vel)
        
        # 仅在刚起跳时计算奖励
        return vel_error * just_took_off * -1.0 # 惩罚误差

    def _reward_air_time(self):
        # 奖励腾空时间 (与目标高度相关)
        target_air_time = torch.sqrt(8 * self.commands[:, 0] / 9.81) # t = 2 * v / g
        air_time_error = torch.square(self.air_time_buf - target_air_time)
        
        # 仅在空中时计算
        return air_time_error * self.in_air * -1.0 # 惩罚误差

    def _reward_landing_dist(self):
        # (Atanassov灵感)
        # 这个奖励只在落地瞬间被计算并缓存在 self.landing_dist_rew 中
        return self.landing_dist_rew

    # (OmniNet灵感)
    def _reward_aerial_posture(self):
        # 仅在空中时惩罚
        pose_error = torch.sum(torch.square(self.dof_pos - self.aerial_tuck_angles), dim=1)
        return pose_error * self.in_air.float()

    # --- 保留SATA的约束奖励 ---
    
    # 继承自 GO2Torque
    # def _reward_soft_dof_pos_limits(self):
    
    # 继承自 GO2Torque
    # def _reward_motor_fatigue(self):
    
    # 继承自 LeggedRobot
    # def _reward_dof_acc(self):
    
    # 继承自 LeggedRobot
    # def _reward_torques(self):