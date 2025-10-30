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

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2JumpCfg(GO2RoughCfg):
    class env(GO2RoughCfg.env):
        # 观测空间 = 60 (SATA基础) - 3 (SATA指令) + 2 (跳跃指令) = 59
        num_observations = 59
        num_actions = 12
        episode_length_s = 4.0 # 缩短episode时长以适应跳跃任务

    class init_state(GO2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.3]  # 初始位置
        
        # 初始姿态：修改为蹲伏（Crouch）姿态，为起跳做准备
        default_joint_angles = { 
            'FL_hip_joint': 0.1, 
            'RL_hip_joint': 0.1,  
            'FR_hip_joint': -0.1, 
            'RR_hip_joint': -0.1, 

            'FL_thigh_joint': 1.0,  # 蹲伏
            'RL_thigh_joint': 1.0,  # 蹲伏
            'FR_thigh_joint': 1.0,  # 蹲伏
            'RR_thigh_joint': 1.0,  # 蹲伏

            'FL_calf_joint': -2.0,   # 蹲伏
            'RL_calf_joint': -2.0,   # 蹲伏
            'FR_calf_joint': -2.0,   # 蹲伏
            'RR_calf_joint': -2.0,   # 蹲伏
        }

    class asset(GO2RoughCfg.asset):
        # 继承SATA的力矩控制URDF
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf' #
        self_collisions = 0
        terminate_after_contacts_on = ["base", "Head", "thigh", "calf"] # 身体碰撞地面则终止
        penalize_contacts_on = []

    class terrain:
        mesh_type = 'plane' # 始终在平地上训练跳跃
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = False # 不使用地形课程，我们使用SATA的生长模型进行任务课程
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False # 平地不需要测量高度

    class commands(GO2RoughCfg.commands):
        heading_command = False
        resampling_time = 3.5 # 每次episode快结束时重新采样

        # (灵感来自 Atanassov et al. 和 OmniNet)
        class ranges:
            target_height = [0.1, 0.5]  # [m] 目标跳跃高度 (垂直跳跃)
            target_forward_dist = [0.0, 0.6] # [m] 目标前向跳跃距离 (前向跳跃)

    class control(GO2RoughCfg.control):
        control_type = 'TG' # 'T': torque control, 'TG': torque control with growth
        activation_process = True
        
        # 关键修改：关闭Hill模型以实现爆发力
        # SATA的Hill模型在高速时会抑制力矩，不利于跳跃
        hill_model = False 
        
        motor_fatigue = True # 保留SATA的疲劳模型
        action_scale = 5
        decimation = 1

    class noise:
        # 复制自SATA
        add_noise = True
        noise_level = 1.0  # (为跳跃任务修改了值)

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.2
            height_measurements = 0.1
            fatigue = 0.5 
        
    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        # (灵感来自 OmniNet)
        # 预定义的空中"收腿"姿态 [rad]
        aerial_tuck_angles = { 
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 1.8,
            'RL_thigh_joint': 1.8,
            'FR_thigh_joint': 1.8,
            'RR_thigh_joint': 1.8,
            'FL_calf_joint': -2.7,
            'RL_calf_joint': -2.7,
            'FR_calf_joint': -2.7,
            'RR_calf_joint': -2.7,
        }

        # 彻底替换SATA的行走奖励
        class scales:
            # 存活与稳定
            termination = -200.0
            stability = -10.0       # 惩罚 roll/pitch 姿态 (projected_gravity)
            tracking_ang_vel = -0.5 # 惩罚 roll/pitch 角速度

            # 跳跃核心奖励
            jump_z_vel = 15.0       # 奖励Z轴起跳速度
            air_time = 15.0         # 奖励腾空时间
            landing_dist = -10.0    # 惩罚落地位置与目标x坐标的误差
            
            # (OmniNet灵感) 空中姿态
            aerial_posture = -1.5   # 惩罚空中姿态与"收腿"姿态的差异
            
            # 消耗与约束
            dof_acc = -1e-6
            motor_fatigue = -0.01   # (SATA)
            soft_dof_pos_limits = -5.0
            torques = -0.0001
            

    class domain_rand:
        # 复制自SATA
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 5.]

        # --- (以下是缺失的属性) ---
        shifted_com_range_x = [-0.2, 0.2]
        shifted_com_range_y = [-0.1, 0.1]
        shifted_com_range_z = [-0.1, 0.1]
        # --- (缺失属性结束) ---

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5 # (这是我们为跳跃任务的修改)
        max_push_vel_ang = 1.0 # (这也是缺失的属性)

        # (SATA父类需要的属性)
        loss_action_obs = True
        loss_rate = 0.1

    class growth:
        # 重用SATA的生长模型
        # 但将其用于调度跳跃难度
        
        # 物理参数课程 (与SATA相同)
        max_torque_scale = 1.0
        start_torque_scale = 0.3
        max_rear_torque_scale = 1.0
        start_rear_torque_scale = 1.0
        max_freq = 200
        start_freq = 100

        # Gompertz 曲线参数 (与SATA相同)
        k = 0.00003
        x0 = 1000 * 24

        # (Atanassov灵感)
        # 课程阶段切换点 (G(t) > 0.5 时，开始学习向前跳)
        forward_jump_threshold = 0.5 

    class test():
        use_test = False


class GO2JumpCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'SATA_Jump' # 修改实验名称
        max_iterations = 5000 # 跳跃可能需要更多训练次数
    
    class algorithm(GO2RoughCfgPPO.algorithm):
        entropy_coef = 0.01 # 鼓励探索