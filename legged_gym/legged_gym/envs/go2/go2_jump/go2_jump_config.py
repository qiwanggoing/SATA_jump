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
        num_observations = 59
        num_actions = 12
        episode_length_s = 4.0 

    class init_state(GO2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.3]  # 初始位置
        
        # (借鉴 Bilibili-go2)
        # 保持蹲伏（Crouch）姿态
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf'
        self_collisions = 0
        terminate_after_contacts_on = ["base", "Head", "thigh", "calf"] 
        penalize_contacts_on = []

    class terrain:
        mesh_type = 'plane' 
        curriculum = False 
        measure_heights = False 

        # --- (修复：添加父类 所需的缺失属性) ---
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # --- (修复结束) ---

        # (SATA行走任务的trimesh参数，我们用不到，但保留在注释中以备将来使用)
        # horizontal_scale = 0.1
        # vertical_scale = 0.005
        # border_size = 25
        # terrain_proportions = [0.2, 0.8, 0, 0, 0.0]
        # slope_treshold = 0.75

    class commands(GO2RoughCfg.commands):
        heading_command = False
        resampling_time = 3.5 

        class ranges:
            target_height = [0.1, 0.5]  
            target_forward_dist = [0.0, 0.6] 

    class control(GO2RoughCfg.control):
        control_type = 'TG' # (SATA) 力矩 + 生长
        activation_process = True
        
        # (SATA) 关闭Hill模型以实现爆发力
        hill_model = False 
        
        motor_fatigue = True 
        action_scale = 5
        decimation = 1

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.2
            height_measurements = 0.1
            fatigue = 0.5 # (SATA) 
            
    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9 # (SATA)
        
        # (OmniNet灵感) 空中"收腿"姿态
        aerial_tuck_angles = { 
            'FL_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RR_hip_joint': 0.0,
            'FL_thigh_joint': 1.8, 'RL_thigh_joint': 1.8, 'FR_thigh_joint': 1.8, 'RR_thigh_joint': 1.8,
            'FL_calf_joint': -2.7, 'RL_calf_joint': -2.7, 'FR_calf_joint': -2.7, 'RR_calf_joint': -2.7,
        }

        # (修复：采用 Bilibili-go2 奖励逻辑)
        class scales:
            # 存活与稳定 (在go2_jump.py中乘以 general_scale)
            termination = -50.0       
            stability = -5.0        
            tracking_ang_vel = -0.5 
            landing_dist = -10.0
            aerial_posture = -1.5   
            
            # --- 核心正向奖励 (不乘以 general_scale) ---
            stand = 10.0            # (新) 奖励回到蹲伏姿态
            jump_z_vel = 20.0       # (新) 奖励Z轴正速度
            air_time = 20.0         # (新) 奖励腾空
            
            # 消耗与约束
            dof_acc = -1e-6
            motor_fatigue = -0.01
            soft_dof_pos_limits = -5.0

    class domain_rand:
        # (SATA)
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 5.]
        shifted_com_range_x = [-0.2, 0.2]
        shifted_com_range_y = [-0.1, 0.1]
        shifted_com_range_z = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5 
        max_push_vel_ang = 1.0 
        loss_action_obs = True
        loss_rate = 0.1

    class growth:
        # (SATA)
        max_torque_scale = 1.0
        start_torque_scale = 0.3
        max_rear_torque_scale = 1.0
        start_rear_torque_scale = 1.0
        max_freq = 200
        start_freq = 100
        k = 0.00003
        x0 = 1000 * 24

        # (Atanassov灵感)
        forward_jump_threshold = 0.5 

    class test:
        use_test = False
        checkpoint = 3000 
        vel = torch.tensor([0.0, 0.0], dtype=torch.float32) # 2D

class GO2JumpCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'SATA_Jump' 
        max_iterations = 5000 
    
    class algorithm(GO2RoughCfgPPO.algorithm):
        entropy_coef = 0.01