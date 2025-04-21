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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.cost_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
            if 'cost' in key:
                self.cost_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
        self.cost_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 4
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # # plot base vel x
        a = axs[0, 2]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[1, 2]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[2, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot torques
        for i in range(3):
            a = axs[i, 0]
            if log["torques"]:
                torques = np.array(log["torques"])
                a.plot(time, torques[:, i], label=f'torque {i}')
                a.set(xlabel='time [s]', ylabel='Torque [Nm]', title='Joint Torques')
                a.legend()
            a = axs[i, 1]
            if log["dof_vels"]:
                dof_vels = np.array(log["dof_vels"])
                a.plot(time, dof_vels[:, i], label=f'vel {i}')
                a.set(xlabel='time [s]', ylabel='Joint vel [rad/s]', title='Joint Velocities')
                a.legend()
        for i in range(4):
            a = axs[i, 3]
            if log["contact_forces_z"]:
                contact_forces_z = np.array(log["contact_forces_z"])
                a.plot(time, contact_forces_z[:, i], label=f'contact force {i}')
                a.set(xlabel='time [s]', ylabel='Contact force [N]', title='Contact Forces')
                a.legend()
        for i in range(4):
            a = axs[3, 0]
            if log["foot_z"]:
                foot_z = np.array(log["foot_z"])
                a.plot(time, foot_z[:, i], label=f'foot z {i}')
                a.set(xlabel='time [s]', ylabel='Foot z [m]', title='Foot z')
                a.legend()
        plt.show()

        nb_rows = 2
        nb_cols = 1
        fig1, axs1 = plt.subplots(nb_rows, nb_cols)
        foot_contact = np.array(log["contact_forces_z"])
        foot_contact_intervals = {
            'FL': foot_contact[:, 0] > 1,
            'FR': foot_contact[:, 1] > 1,
            'RL': foot_contact[:, 2] > 1,
            'RR': foot_contact[:, 3] > 1,
        }
        for i, (foot, contact) in enumerate(foot_contact_intervals.items()):
            start_idx = None
            for j, state in enumerate(contact):
                if state and start_idx is None:
                    start_idx = j
                elif not state and start_idx is not None:
                    end_idx = j - 1
                    start_time = time[start_idx]
                    end_time = time[end_idx]
                    axs1[0].plot([start_time, end_time], [i, i], lw=8, color=f"C{i}")
                    start_idx = None
            if start_idx is not None:
                start_time = time[start_idx]
                end_time = time[-1]
                axs1[0].plot([start_time, end_time], [i, i], lw=8, color=f"C{i}")

        # 设置标签和样式
        foot_labels = ["FL", "FR", "RL", "RR"]
        axs1[0].set_yticks(range(len(foot_labels)))
        axs1[0].set_yticklabels(foot_labels)
        axs1[0].set_xlim(time[0], time[-1])
        axs1[0].set_ylim(-1, 4)
        axs1[0].set_xlabel("Time (s)")
        axs1[0].set_ylabel("Feet")
        axs1[0].spines['top'].set_visible(False)
        axs1[0].spines['right'].set_visible(False)
        plt.show()

        # plot rewards
        fig2, axs2 = plt.subplots(1, 1)
        if log["reward"]:
            rewards = np.array(log["reward"])
            np.save('rewards.npy', rewards)
            np.save('time.npy', time)
            cumulative_rewards = np.cumsum(rewards)
            axs2.plot(time, cumulative_rewards, label='cumulative_rewards')
            axs2.set(xlabel='time [s]', ylabel='cumulative_rewards', title='cumulative_rewards')
            axs2.legend()
        plt.show()


    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print('-------------------------------------------')
        print("Average costs per second:")
        for key, values in self.cost_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()