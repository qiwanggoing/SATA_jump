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

# 导入SATA项目的基础配置 (GO2)
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2JumpCfg(GO2RoughCfg):
    
    # --- 移植自 "Jumping" 项目：课程设置 ---
    curriculum_thresholds = [0.8, 0.8, 0.7] #

    class obstacles:
        track_obstacle_height = True
        track_obstacle_width = True
        obstacle_height_range = [0.0, 0.25]
        obstacle_width_range = [0.0, 0.4]
    # ----------------------------------------
    
    class env(GO2RoughCfg.env):
        # --- 融合修改：观测维度 ---
        # SATA(60) - 基础(3)指令 + 额外SATA(24) + 跳跃任务(10) = 67
        # 基础(3+3+3+12+12=33) + 额外SATA(12 torques + 12 fatigue = 24) + 跳跃任务(10) = 67
        num_observations = 67
        num_actions = 12
        episode_length_s = 10 #

    class init_state(GO2RoughCfg.init_state):
        # --- 保留SATA的初始姿态 ---
        pos = [0.0, 0.0, 0.10]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 1.45,  # [rad]
            'RL_thigh_joint': 1.45,  # [rad]
            'FR_thigh_joint': 1.45,  # [rad]
            'RR_thigh_joint': 1.45,  # [rad]

            'FL_calf_joint': -2.5,  # [rad]
            'RL_calf_joint': -2.5,  # [rad]
            'FR_calf_joint': -2.5,  # [rad]
            'RR_calf_joint': -2.5,  # [rad]
        } #

    class asset(GO2RoughCfg.asset):
        # --- 关键：保留SATA的URDF ---
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf' #
        self_collisions = 0
        terminate_after_contacts_on = ["Head"]
        penalize_contacts_on = ["thigh", "calf", "trunk"]
        penalize_contacts_force = 100.0

    class terrain(GO2RoughCfg.terrain):
        # --- 保留SATA的地面设置 ---
        mesh_type = 'trimesh'
        # ... (SATA的其余地形参数) ...
        terrain_proportions = [0.2, 0.8, 0, 0, 0.0]
        slope_treshold = 0.75
    
    # --- 移植自 "Jumping" 项目：任务指令 ---
    class commands(GO2RoughCfg.commands):
        curriculum_profile = 'jump'
        resampling_time = 5
        class ranges:
            target_lin_x = [0.0, 0.0]
            target_lin_y = [0.0, 0.0]
            target_ang_yaw = [0.0, 0.0]
            target_height = [0.4, 0.4]
    #
    # ----------------------------------------
    
    class control(GO2RoughCfg.control):
        # --- 关键：保留SATA的“肌肉” (力矩控制) ---
        control_type = 'TG' # 'T': torque control, 'TG': torque control with growth
        activation_process = True
        hill_model = True
        motor_fatigue = True
        action_scale = 5
        decimation = 1
    #
    # ----------------------------------------
    
    class noise(GO2RoughCfg.noise):
        # --- 保留SATA的噪声设置 ---
        add_noise = True
        noise_level = 1.5
        class noise_scales(GO2RoughCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.2
            height_measurements = 0.1
            fatigue = 0.5
    #
    # ----------------------------------------
    
    class rewards(GO2RoughCfg.rewards):
        # --- 移植自 "Jumping" 项目：奖励基础设置 ---
        clip_reward = 0.1
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        # ----------------------------------------
        
        class scales:
            # --- 移植自 "Jumping" 项目：21个奖励项 ---
            # Task rewards
            jump = 10.0
            landing_pose = 5.0
            orientation = 5.0
            ang_vel_z = 5.0

            # Penalties
            termination = -10.0
            collision = -10.0
            base_height_binary = -10.0
            lin_vel_z = -5.0
            ang_vel_xy = -0.05
            orientation = -5.0
            stand_still = -0.5

            # Smoothness & Energy
            torques = -1e-05
            dof_vel = -0.001
            dof_acc = -2.5e-07
            action_rate = -0.01
            dof_pos_limits = -5.0

            # Leg & Gait Shaping
            hop_stand_still = -0.5
            swing_curled = -0.5
            swing_unjerk = 0.3
            stance_unjerk = 0.3
            feet_air_time = 0.5
        #
        # ----------------------------------------

    class domain_rand(GO2RoughCfg.domain_rand):
        # --- 保留SATA的域随机化 ---
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 5.]
        shifted_com_range_x = [-0.2, 0.2]
        shifted_com_range_y = [-0.1, 0.1]
        shifted_com_range_z = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 1.5
        max_push_vel_ang = 1.0
        loss_action_obs = True
        loss_rate = 0.1
    #
    # ----------------------------------------
    
    class growth:
        # --- 关键：保留SATA的“生长” (物理发育课程) ---
        max_torque_scale = 1.0
        start_torque_scale = 0.3
        max_rear_torque_scale = 1.0
        start_rear_torque_scale = 1.0

        max_freq = 200
        start_freq = 100

        k = 0.00003
        x0 = 1000 * 24
    #
    # ----------------------------------------

    class test:
        # --- 保留SATA的测试设置 ---
        use_test = False
        checkpoint = 3000
        vel = torch.tensor([1.0, 0.0, 0.0, 0.], dtype=torch.float32)
    #
    # ----------------------------------------

class GO2JumpCfgPPO(GO2RoughCfgPPO):
    
    class runner(GO2RoughCfgPPO.runner):
        # --- 保留SATA的Runner设置 ---
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'SATA_Jump' # 修改实验名称
        max_iterations = 3000 # 你可能需要增加这个值
    #
    # ----------------------------------------
    
    # --- 移植自 "Jumping" 项目：更深的策略网络 ---
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    #
    # ----------------------------------------