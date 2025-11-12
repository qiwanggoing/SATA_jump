# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import torch
# 导入 GO2RoughCfg 作为我们的基础，因为它有正确的关节名称
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
# 导入 LeggedRobotCfg 以便引用基类配置
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# --- 这是我们的新跳跃任务配置 ---
class GO2TorqueJumpCfg(GO2RoughCfg):
    
    class env(GO2RoughCfg.env):
        # [核心修改] 移植跳跃的观测和指令维度
        # (基于 curriculum-quadruped-jumping-drl 的 1014 维)
        num_observations = 1014 
        num_actions = 12
        num_commands = 13 # (跳跃任务需要13维指令)
        episode_length_s = 4.0 # (跳跃是短时任务)

    class init_state(GO2RoughCfg.init_state):
        # [核心修改] 跳跃从站立姿态开始，而不是SATA的趴下姿态
        # (我们直接使用 GO2RoughCfg 的站立姿态)
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = GO2RoughCfg.init_state.default_joint_angles # (继承站立姿态)

    class observations:
        # 5. [核心修改] 移植跳跃的观测历史配置
        # (完整复制 go1_upwards_config.py 中的 class observations)
        use_contact_history = False # (注意：跳跃仓库使用的是 state_history，SATA/legged_gym用的是 contact_history)
        use_state_history = True
        state_history_length = 20
        history_gait_check = False
        fixed_phase = False
        use_fixed_phase = False
        phase_jumping_target = False
        jumping_target = True
        known_quaternion = True
        known_contact_states = False
        known_contact_forces = False
        known_contact_normals = False
        known_contact_feet = True
        known_height = False
        known_error_quaternion = False
        known_ori_error = False
        pass_has_jumped = True
        pass_jumping_command = False

    class asset(GO2RoughCfg.asset):
        # (保持SATA的力控URDF)
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf'
        self_collisions = 0
        terminate_after_contacts_on = ["base", "thigh"] # (同跳跃仓库)
        penalize_contacts_on = ["thigh", "calf"]

    class terrain:
        # [核心修改] 为了专注于跳跃，我们先使用平面
        mesh_type = 'plane'
        measure_heights = False # 平面不需要

        # [修复] 为SATA基类 添加占位符属性
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        # (以下属性是为 'trimesh' 准备的，但SATA基类 也需要它们)
        curriculum = False 
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0] # (任意值)
        max_init_terrain_level = 0 # (任意值)

    class commands(GO2RoughCfg.commands):
        # (跳跃指令在 env/observations 中定义，这里可以简化)
        curriculum = False
        resampling_time = 4.0 # (整个回合使用同一组指令)
        class ranges:
            # (这些是SATA的，但跳跃的13维指令在 _resample_commands 中生成)
            # (我们暂时保留它，但 _resample_commands 会覆盖它)
            lin_vel_x = [0.0, 0.0]  # 原地跳跃
            lin_vel_y = [0.0, 0.0]  # 原地跳跃
            ang_vel_yaw = [0.0, 0.0] # 原地跳跃

    class control(GO2RoughCfg.control):
        # 1. [核心修改] 设置为纯力矩控制
        control_type = 'T' 
        
        # 2. [核心修改] 禁用SATA的生物力学模型
        activation_process = False
        hill_model = False
        motor_fatigue = False
        
        # 3. [核心修改] 设置Action Scale为最大力矩
        # Go2电机的最大力矩在 go2_torque.urdf 中是 23.5 Nm
        action_scale = 23.5 
        
        # 4. [核心修改] 禁用PD控制器
        stiffness = {'joint': 0.} 
        damping = {'joint': 0.}
        
        # 5. [核心修改] 匹配跳跃仓库的控制频率
        decimation = 4 # (Sim 200Hz / Control 50Hz = 4)

    class noise:
        # (我们禁用了噪声，专注于任务本身)
        add_noise = False 
        noise_level = 1.0 # (SATA基类 需要这个)

        # [修复] 为SATA基类 添加占位符子类
        # (这些值来自SATA的 legged_robot_config.py)
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
        
    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False 

        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100.
        base_height_target = 0.35 

        default_pose_tracking_sigma = 0.25   
        command_pos_tracking_sigma = 0.25    
        command_ori_tracking_sigma = 0.25      
        height_tracking_sigma = 0.25           
        post_landing_pos_tracking_sigma = 0.25  


        class scales:
            # (这是我们之前复制的权重)
            termination = -20.0
            task_pos = 200.0
            task_ori = 200.0
            task_max_height = 2000.0 
            base_height_flight = 80.0
            base_height_stance = 5.0 
            jumping = 50.0
            post_landing_pos = 3.0
            post_landing_ori = 3.0
            default_pose = 6.0
            feet_distance = -20.0 
            action_rate = -0.2
            dof_acc = -1e-6
            early_contact = 5.0

            # [推荐] 力控稳定项 (这些不需要额外参数)
            orientation = -5.0 
            ang_vel_xy = -0.05 
            lin_vel_z = -5.0

    class domain_rand:
        # [核心修改] 禁用所有域随机化，先让力控学会基本任务
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False
        loss_action_obs = False
        push_interval_s = 15.0
    # 7. [核心修改] 移除SATA的生长模型
    # (删除了整个 class growth)

class GO2TorqueJumpCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        experiment_name = 'go2_torque_jump'
        max_iterations = 15000 # 跳跃任务可能需要更长的训练时间
    class algorithm(GO2RoughCfgPPO.algorithm):
        entropy_coef = 0.01