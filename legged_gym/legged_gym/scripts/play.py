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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 20
    env_cfg.control.control_type = 'T'
    
    # --- (修复：适配 go2_jump 任务) ---
    # 检查任务是否为 'go2_jump'
    is_jump_task = (args.task == 'go2_jump')

    if is_jump_task:
        print("INFO: 'go2_jump' 任务检测到。正在修改 play.py 脚本以适配2D跳跃指令。")
        env_cfg.test.use_test = True # 强制使用测试模式
        # (修复：使用2D指令 (h_cmd, fwd_cmd) 替换 4D行走指令)
        env_cfg.test.vel = torch.tensor([0.3, 0.2], dtype=torch.float32) # (h=0.3m, fwd=0.2m)
        env_cfg.commands.heading_command = False # (修复：跳跃任务不需要heading)
    else:
        # (SATA行走任务的原始逻辑)
        env_cfg.test.use_test = True
        env_cfg.test.checkpoint = 3000
        env_cfg.test.vel = torch.tensor([0.0, 0.0, 0., 0.0], dtype=torch.float32)
        env_cfg.commands.heading_command = True
    # --- (修复结束) ---

    env_cfg.control.activation_process = True
    env_cfg.control.hill_model = True
    env_cfg.control.motor_fatigue = True
    
    env_cfg.terrain.mesh_type = 'plane'  # 'trimesh'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.terrain.terrain_proportions = [0, 1, 0, 0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 10  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    img_idx = 0
    
    # --- (修复：使CHANGE_VEL逻辑适配跳跃任务) ---
    vel_h = 0.3 # 目标高度
    vel_f = 0.0 # 目标前向距离
    change_vel_h = 0.1
    change_vel_f = 0.1
    
    # (SATA的行走任务速度)
    # vel_x = 1.0
    # change_vel = 0.2
    
    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        foot_z = env.rigid_body_states[0, env.feet_indices, 2].cpu().numpy()
        
        if CHANGE_VEL and is_jump_task:
            # (跳跃任务的指令切换逻辑)
            if i % 100 == 0:
                # 切换高度
                if vel_h > 0.45 or vel_h < 0.15:
                    change_vel_h = -change_vel_h
                vel_h += change_vel_h
                
                # 切换前向距离 (仅在课程的后半段)
                if env.general_scale > env_cfg.growth.forward_jump_threshold:
                     if vel_f > 0.5 or vel_f < 0.0:
                        change_vel_f = -change_vel_f
                     vel_f += change_vel_f
                
                env.commands[0, 0] = vel_h
                env.commands[0, 1] = vel_f
                env_cfg.test.vel = torch.tensor([vel_h, vel_f], dtype=torch.float32)

        elif CHANGE_VEL and not is_jump_task:
            # (SATA行走任务的原始逻辑)
            if i % 100 == 0:
                vel_x = 1.0 # 示例：固定行走速度
                env_cfg.test.vel = torch.tensor([vel_x , 0.0, 0., 0.], dtype=torch.float32)
        # --- (修复结束) ---

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            camera_position = robot_pos + np.array([1, 1, 1])
            env.set_camera(camera_position, robot_pos)
            
        if i < stop_state_log:
            logger.log_states(
                {
                    'actions': actions[robot_index].detach().cpu().numpy(),
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    
                    # --- (修复：适配 go2_jump 任务) ---
                    # (SATA行走任务的原始日志)
                    # 'command_x': env.commands[robot_index, 0].item(),
                    # 'command_y': env.commands[robot_index, 1].item(),
                    # 'command_yaw': env.commands[robot_index, 2].item(),
                    
                    # (go2_jump 任务的新日志)
                    'command_height': env.commands[robot_index, 0].item(),
                    'command_fwd_dist': env.commands[robot_index, 1].item(),
                    # --- (修复结束) ---
                    
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_height': env.root_states[robot_index, 2].item(),
                    'torques': env.torques[robot_index].cpu().numpy(),
                    'dof_vels': env.dof_vel[robot_index].cpu().numpy(),
                    'foot_z': foot_z,
                    'reward': rews[robot_index].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    
    # --- (修复：默认关闭 CHANGE_VEL，因为它在跳跃时可能不稳定) ---
    CHANGE_VEL = False
    
    args = get_args()
    play(args)