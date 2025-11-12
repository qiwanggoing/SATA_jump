from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_torque_jump_config import GO2TorqueJumpCfg # 导入我们修改后的新配置
from legged_gym.utils.math import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from collections import deque
import numpy as np
import os # <-- [修复1] 添加此行
from legged_gym import LEGGED_GYM_ROOT_DIR # <-- [修复2] 添加此行

class Go2TorqueJump(LeggedRobot):
    """
    Go2TorqueJump 类:
    - 继承自 SATA 仓库的 LeggedRobot (用于力控基础).
    - 移植了 curriculum-quadruped-jumping-drl 仓库的
      观测、状态和奖励逻辑 (来自 uploaded:legged_robot.py 和 uploaded:go1_upwards_config.py).
    - 禁用了 SATA 的生物力学和生长模型.
    """

    def __init__(self, cfg: GO2TorqueJumpCfg, sim_params, physics_engine, sim_device, headless):
        
        # [核心移植 1] 移植跳跃的状态变量和历史缓冲区
        # 来源: uploaded:legged_robot.py (L550-L558)
        # -----------------------------------------------------------------
        self.state_history_length = cfg.observations.state_history_length
        self.state_history_buffer_sh = torch.zeros(cfg.env.num_envs,
                                                self.state_history_length,
                                                42, # 3+3+12+12+12 = 42
                                                dtype=torch.float,
                                                device=sim_device,
                                                requires_grad=False)
        self.state_history_buffer = self.state_history_buffer_sh.clone()

        self.mid_air = torch.zeros(cfg.env.num_envs, 
                                   dtype=torch.bool,
                                   device=sim_device,
                                   requires_grad=False)
        self.has_jumped = torch.zeros(cfg.env.num_envs,
                                      dtype=torch.bool,
                                      device=sim_device,
                                      requires_grad=False)
        self.max_height = torch.zeros(cfg.env.num_envs,
                                      dtype=torch.float,
                                      device=sim_device,
                                      requires_grad=False)
        self.landing_poses = torch.zeros(cfg.env.num_envs,
                                         7, # (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
                                         dtype=torch.float,
                                         device=sim_device,
                                         requires_grad=False)
        
        # (跳跃仓库 中的其他历史缓冲区)
        self.quat_history = torch.zeros(cfg.env.num_envs,
                                        self.state_history_length,
                                        4,
                                        dtype=torch.float,
                                        device=sim_device,
                                        requires_grad=False)
        self.contact_history = torch.zeros(cfg.env.num_envs,
                                           self.state_history_length,
                                           4,
                                           dtype=torch.float,
                                           device=sim_device,
                                           requires_grad=False)
        self.contact_history_sh = self.contact_history.clone()
        # -----------------------------------------------------------------
        
        # 调用SATA仓库的 LeggedRobot.__init__
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 移植 curriculum-jumping 缺失的 rigid_body_states 缓冲区
        # _reward_feet_distance 依赖此缓冲区
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
    
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], feet_names[i], gymapi.DOMAIN_SIM)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for i in range(len(penalized_contact_names)):
            # [核心修复] 必须使用 ..._index
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0],
                                                                                    self.actor_handles[0],
                                                                                    penalized_contact_names[i],
                                                                                    gymapi.DOMAIN_SIM)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                   device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            # [核心修复] 必须使用 ..._index
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0],
                                                                                            self.actor_handles[0],
                                                                                            termination_contact_names[i],
                                                                                            gymapi.DOMAIN_SIM)
    def step(self, actions):
        """ 
        [核心修复] 重写基类 的 step()
        以添加 curriculum-jumping 所需的
        `refresh_rigid_body_state_tensor`。
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
            # [核心修复] 添加此行 (SATA 基类中缺失)
            self.gym.refresh_rigid_body_state_tensor(self.sim) 

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        # 调用SATA基类的 reset_idx
        super().reset_idx(env_ids)

        # [核心移植 2] 移植跳跃状态的重置
        # 来源: uploaded:legged_robot.py (L675-L680)
        # -----------------------------------------------------------------
        self.state_history_buffer[env_ids] = 0.
        self.state_history_buffer_sh[env_ids] = 0.
        self.mid_air[env_ids] = False
        self.has_jumped[env_ids] = False
        self.max_height[env_ids] = 0.
        self.landing_poses[env_ids] = 0.
        self.quat_history[env_ids] = 0.
        self.contact_history[env_ids] = 0.
        self.contact_history_sh[env_ids] = 0.
        # -----------------------------------------------------------------

    def _post_physics_step_callback(self):
        # [核心移植 3] 移植跳跃状态的更新
        # 来源: uploaded:legged_robot.py (L752-L756)
        # -----------------------------------------------------------------
        # (注意：self.contacts 依赖于 self.contact_forces)
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        
        self.mid_air = torch.all(torch.logical_not(self.contacts), dim=-1)
        self.has_jumped = torch.logical_or(self.has_jumped, self.mid_air)
        
        # 记录最大高度
        self.max_height = torch.where(torch.logical_and(self.has_jumped, self.root_states[:, 2] > self.max_height), self.root_states[:, 2], self.max_height)
        
        # 记录落地姿态 (在从空中切换到非空中的瞬间)
        landing_ids = (torch.logical_and(self.has_jumped, torch.logical_not(self.mid_air)) & torch.logical_not(torch.all(self.landing_poses == 0, dim=-1))).nonzero(as_tuple=False).flatten()
        if len(landing_ids) > 0:
            self.landing_poses[landing_ids] = self.root_states[landing_ids, :7]
        # -----------------------------------------------------------------

        # 调用SATA基类的回调函数 (这将调用 _resample_commands)
        # 来源: qiwanggoing/sata_jump/SATA_jump-957737329133e6842c89c14999bb1fc47703e724/legged_gym/legged_gym/envs/base/legged_robot.py (L369)
        super()._post_physics_step_callback()

    def _resample_commands(self, env_ids):
        # [核心移植 4] 移植13维指令的生成
        # 来源: uploaded:legged_robot.py (L758-L768)
        # -----------------------------------------------------------------
        if self.cfg.commands.curriculum == "reference_gait":
            # (此分支在 go1_upwards_config.py 中未用，但我们完整移植)
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["body_height_cmd"][0], self.command_ranges["body_height_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["step_frequency_cmd"][0], self.command_ranges["step_frequency_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["gait"][0], self.command_ranges["gait"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 3:6] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 3), device=self.device)
            self.commands[env_ids, 6:9] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 3), device=self.device)
            self.commands[env_ids, 9:12] = torch_rand_float(self.command_ranges["footswing_height_cmd"][0], self.command_ranges["footswing_height_cmd"][1], (len(env_ids), 3), device=self.device)
        elif self.cfg.commands.curriculum == "no_curriculum_no_ref":
             # (这是 go1_upwards_config.py 使用的逻辑)
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["body_height_cmd"][0], self.command_ranges["body_height_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["step_frequency_cmd"][0], self.command_ranges["step_frequency_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 7] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 9] = torch_rand_float(self.command_ranges["footswing_height_cmd"][0], self.command_ranges["footswing_height_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 12] = torch_rand_float(self.command_ranges["jump_height_cmd"][0], self.command_ranges["jump_height_cmd"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # -----------------------------------------------------------------

    def compute_observations(self):
        # [核心移植 5] 移植1014维观测的实现
        # 来源: uploaded:legged_robot.py (L911-L931)
        # (这是一个复杂函数，我们完整复制)
        # -----------------------------------------------------------------
        # 1. 构建当前 42 维状态
        current_state = torch.cat((self.base_lin_vel,
                                   self.base_ang_vel,
                                   self.dof_pos - self.default_dof_pos,
                                   self.dof_vel,
                                   self.actions), dim=-1)
        # 2. 更新历史缓冲区
        self.state_history_buffer = torch.cat((self.state_history_buffer[:, 1:], current_state.unsqueeze(1)), dim=1)
        self.state_history_buffer_sh = self.state_history_buffer.clone()

        # 3. 复制跳跃仓库 的其他历史和拼接逻辑
        self.quat_history = torch.cat((self.quat_history[:, 1:], self.base_quat.unsqueeze(1)), dim=1)
        self.contact_history = torch.cat((self.contact_history[:, 1:], self.contacts.unsqueeze(1)), dim=1)
        self.contact_history_sh = self.contact_history.clone()
        
        self.obs_buf_sh = self.state_history_buffer_sh.reshape(self.num_envs, -1) # 42 * 20 = 840 维
        
        # 4. 根据 config 拼接
        if self.cfg.observations.use_state_history:
            obs_buf_augmented = self.obs_buf_sh
        if self.cfg.observations.known_quaternion:
            quat_history_sh = self.quat_history.reshape(self.num_envs, -1) # 4 * 20 = 80 维
            obs_buf_augmented = torch.cat((obs_buf_augmented, quat_history_sh), dim=-1)
        if self.cfg.observations.known_contact_feet:
            contact_history_sh = self.contact_history_sh.reshape(self.num_envs, -1) # 4 * 20 = 80 维
            obs_buf_augmented = torch.cat((obs_buf_augmented, contact_history_sh), dim=-1)
        if self.cfg.observations.jumping_target:
            obs_buf_augmented = torch.cat((obs_buf_augmented, self.commands), dim=-1) # 13 维
        if self.cfg.observations.pass_has_jumped:
            obs_buf_augmented = torch.cat((obs_buf_augmented, self.has_jumped.unsqueeze(1)), dim=-1) # 1 维
        
        # 5. 将最终的 1014 维向量赋给 self.obs_buf
        self.obs_buf = obs_buf_augmented
        # -----------------------------------------------------------------

    def _compute_torques(self, actions):
        # [核心修改 6] 移植纯力矩控制
        # 来源: `go2_torque_jump_config.py` (L36)
        # -----------------------------------------------------------------
        # actions_scaled 的范围是 [-23.5, 23.5] (或 config 中的 33.5)
        actions_scaled = actions * self.cfg.control.action_scale
        # 直接应用力矩，无生物力学模型
        self.torques = torch.clip(actions_scaled, 
                                 -self.torque_limits, 
                                 self.torque_limits)
        return self.torques
        # -----------------------------------------------------------------
    
    def _reward_task_pos(self):
        # 来源: uploaded:legged_robot.py (L1047-L1054)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.episode_length_buf == self.max_episode_length).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            tracking_error = torch.sum(torch.square(self.root_states[idx,:2] - self.initial_root_states[idx,:2]), dim=-1)
            rew[idx] = torch.exp(-torch.square(tracking_error[idx])/self.cfg.rewards.command_pos_tracking_sigma)
        return rew
    
    def _reward_task_ori(self):
        # 来源: uploaded:legged_robot.py (L1056-L1064)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.episode_length_buf == self.max_episode_length).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            quat_des = self.initial_root_states[idx,3:7] # (目标是初始朝向)
            quat_ini = self.landing_poses[idx,3:7]
            ori_tracking_error = quat_mul(quat_ini, quat_conjugate(quat_des))
            ori_tracking_error_yaw = ori_tracking_error[:, 2] # (只关心Yaw)
            rew[idx] = torch.exp(-torch.square(ori_tracking_error_yaw[idx])/self.cfg.rewards.command_ori_tracking_sigma)
        return rew

    def _reward_post_landing_pos(self):
        # 来源: uploaded:legged_robot.py (L1066-L1073)
        rew = torch.zeros(self.num_envs, device=self.device)
        env_ids = (self.has_jumped).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            tracking_error = torch.sum(torch.square(self.root_states[env_ids,:2] - self.landing_poses[env_ids,:2]), dim=-1)
            rew[env_ids] = torch.exp(-torch.square(tracking_error[env_ids])/self.cfg.rewards.post_landing_pos_tracking_sigma)
        return rew

    def _reward_post_landing_ori(self):
        # 来源: uploaded:legged_robot.py (L1075-L1082)
        rew = torch.zeros(self.num_envs, device=self.device)
        env_ids = (self.has_jumped).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            quat_des = self.landing_poses[env_ids,3:7] # (目标是落地姿态)
            quat_ini = self.root_states[env_ids,3:7]
            ori_tracking_error = quat_mul(quat_ini, quat_conjugate(quat_des))
            rew[env_ids] = torch.exp(-torch.square(ori_tracking_error[env_ids, 0:3]).sum(dim=-1)/self.cfg.rewards.command_ori_tracking_sigma)
        return rew

    def _reward_task_max_height(self):
        # 来源: uploaded:legged_robot.py (L1084-L1090)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.episode_length_buf == self.max_episode_length).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            rew[idx] = torch.exp(-torch.square(self.max_height[idx] - self.commands[idx,12])/self.cfg.rewards.height_tracking_sigma) * self.has_jumped[idx]
        return rew

    def _reward_base_height_flight(self):
        # 来源: uploaded:legged_robot.py (L1092-L1096)
        rew = torch.zeros(self.num_envs, device=self.device)
        rew[self.mid_air] = (self.root_states[self.mid_air, 2] - self.commands[self.mid_air,0]) # (奖励在空中的高度，目标来自指令)
        rew[~self.mid_air] = 0
        return rew

    def _reward_base_height_stance(self):
        # 来源: uploaded:legged_robot.py (L1098-L1103)
        rew = torch.zeros(self.num_envs, device=self.device)
        squat_idx = (~self.mid_air & ~self.has_jumped).nonzero(as_tuple=False).flatten()
        if len(squat_idx) > 0:
            # (奖励下蹲到 0.2m)
            rew[squat_idx] = torch.exp(-torch.square(self.root_states[squat_idx, 2] - 0.20)/self.cfg.rewards.base_height_tracking_sigma)
        return rew

    def _reward_jumping(self):
        # 来源: uploaded:legged_robot.py (L1117-L1123)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.episode_length_buf == self.max_episode_length).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            rew[idx] = (self.has_jumped[idx] & (self.max_height[idx] > 0.5))
        return rew

    def _reward_feet_distance(self):
        # 来源: uploaded:legged_robot.py (L1125-L1146)
        rew = torch.zeros(self.num_envs, device=self.device)
        feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        feet_pos_body_frame = quat_rotate_inverse(self.base_quat.unsqueeze(1).repeat(1,4,1), feet_pos - self.root_states[:,:3].unsqueeze(1).repeat(1,4,1))

        feet_pos_des = torch.zeros(self.num_envs, 4, 3, device=self.device)
        feet_pos_des[:,:,:2] = self.cfg.init_state.rel_foot_pos[:2] # (XY 保持在初始位置)
        feet_pos_des[:,:,2 ] = -0.15 # (Z 轴收腿)

        feet_error = feet_pos_body_frame - feet_pos_des
        
        # [核心] 只在空中 (mid_air) 且高度 > 0.45m 时惩罚
        idx = (self.mid_air & (self.root_states[:,2] > 0.45)).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            rew[idx] = torch.sum(torch.square(feet_error[idx]),dim=-1).sum(dim=-1) # (权重是负的，所以这是惩罚)
        
        return rew

    def _reward_early_contact(self):
        # 来源: uploaded:legged_robot.py (L1148-L1153)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.episode_length_buf < 50).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            rew[idx] = torch.all(self.contacts[idx], dim=-1) # (奖励在回合开始时四足着地)
        return rew

    def _reward_default_pose(self):
        # 来源: uploaded:legged_robot.py (L1155-L1158)
        rew = torch.zeros(self.num_envs, device=self.device)
        idx = (self.has_jumped).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            # (奖励落地后恢复默认站立姿态)
            rew[idx] = torch.exp(-torch.sum(torch.square(self.dof_pos[idx] - self.default_dof_pos[idx]), dim=-1)/self.cfg.rewards.default_pose_tracking_sigma)
        return rew