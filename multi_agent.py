import os
import sys
import shutil
import os.path as osp
import gym
import random
import pickle
import os.path
import math
import glob
import numpy as np

from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import *
from utils.hyperparameters import Config
from utils.plot import plot_reward
from model import DQN

from agents.BaseAgent import BaseAgent
import csv


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_multi(nn.Module):
    def __init__(self, env_list):
        super(DQN_multi, self).__init__()

        self.env_list = env_list

        self.input_shape = env_list[0][1].observation_space.shape
        self.num_actions_list = list()
        for env_name, env, _ in env_list:
            self.num_actions_list.append(env.action_space.n)

        self.conv1 = nn.Conv2d(
            self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)

        fc2_list = list()
        for num_actions in self.num_actions_list:
            fc2 = nn.Linear(512, num_actions)
            fc2_list.append(fc2)
        self.fc2_list = torch.nn.ModuleList(fc2_list)

    def forward(self, x, env_id):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        fc2 = self.fc2_list[env_id]
        x = fc2(x)

        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)


class Multi_Agent(nn.Module):

    def __init__(self, env_list, config, log_dir_list, mode='train'):

        super(Multi_Agent, self).__init__()

        if mode == 'test':
            self.env_list = env_list
            self.device = config.device
            self.model = DQN_multi(self.env_list)
            self.model = self.model.to(self.device)

        elif mode == 'train':
            self.log_dir_list = log_dir_list
            self.env_list = env_list
            self.num_feats_list = [
                env.observation_space.shape for _, env, _ in env_list]
            self.num_actions_list = [
                env.action_space.n for _, env, _ in env_list]

            self.lr = config.LR
            self.device = config.device
            self.gamma = config.GAMMA
            self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
            self.experience_replay_size = config.EXP_REPLAY_SIZE
            self.batch_size = config.BATCH_SIZE
            self.learn_start = config.LEARN_START
            self.update_freq = config.UPDATE_FREQ

            # loss related
            self.loss_type = config.loss_type
            self.kl_tao = config.KL_TAO
            self.kl_ratio = config.KL_ratio
            
            self.update_count = 0

            # load teacher model
            self.load_teacher_model()
            # build model
            self.model = DQN_multi(self.env_list)
            self.target_model = DQN_multi(self.env_list)
            self.target_model.load_state_dict(self.model.state_dict())

            # optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            self.model = self.model.to(self.device)
            self.target_model.to(self.device)
            for teacher_model in self.teacher_model_list:
                teacher_model = teacher_model.to(self.device)

            self.declare_memory()

        else:
            raise ValueError('Invalid mode {}'.format(mode))

    def load_teacher_model(self):
        self.teacher_model_list = list()
        for env_name, env, _ in self.env_list:
            num_feats = env.observation_space.shape
            num_actions = env.action_space.n
            model = DQN(num_feats, num_actions)
            pretrained_weights = osp.join(
                'pretrained_weights', env_name+'.dump')
            model.load_state_dict(torch.load(pretrained_weights))
            for param in model.parameters():
                param.requires_grad = False
            self.teacher_model_list.append(model)

    def load_student_model(self, student_path, env_id):
        env_name = self.env_list[env_id][0]
        student_weights = osp.join(student_path, env_name + '.dump')
        self.model.load_state_dict(torch.load(student_weights))
        print('Student model loaded from {}'.format(student_weights))

    def declare_memory(self):
        self.teacher_memory_list = list()
        self.student_memory_list = list()
        for env_name, t_env, s_env in self.env_list:
            t_memory = ExperienceReplayMemory(self.experience_replay_size)
            self.teacher_memory_list.append(t_memory)
            s_memory = ExperienceReplayMemory(self.experience_replay_size)
            self.student_memory_list.append(s_memory)

    def get_Q(self, s, env_id):
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            teacher_model = self.teacher_model_list[env_id]
            Q = teacher_model(X).cpu().numpy()
            action = np.argmax(Q.squeeze())
            return Q, action

    def get_policy_Q(self, s, env_id):
        """ get Q and action from self.model """
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            Q = self.model(X, env_id).cpu().numpy()
            action = np.argmax(Q.squeeze())
            return Q, action

    def get_action(self, s, env_id, eps):
        with torch.no_grad():
            if np.random.random() >= eps:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X, env_id).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions_list[env_id])

    def prep_minibatch_kd(self, env_id):
        # random transition batch is taken from experience replay memory
        transitions = self.teacher_memory_list[env_id].sample(self.batch_size)
        batch_state, batch_Q = zip(*transitions)
        shape = (-1,) + self.num_feats_list[env_id]
        batch_state = torch.tensor(
            batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_Q = torch.tensor(
            batch_Q, device=self.device, dtype=torch.float)
        batch_Q = batch_Q.detach()
        batch_Q = torch.squeeze(batch_Q)
        return batch_state, batch_Q

    def compute_loss_kd(self, batch_vars, env_id):
        batch_state, teacher_Q = batch_vars
        current_Q = self.model(batch_state, env_id)

        if self.loss_type == 'MSE':
            loss = self.MSE(teacher_Q, current_Q)
        elif self.loss_type == 'KL':
            loss = self.KL(teacher_Q, current_Q)
        else:
            pass
        loss = loss.mean()
        return loss


    def prep_minibatch_dqn(self, env_id):
        # random transition batch is taken from experience replay memory
        transitions = self.student_memory_list[env_id].sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats_list[env_id]

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.uint8)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values
    

    def compute_loss_dqn(self, batch_vars, env_id):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, = batch_vars
        # estimate
        current_q_values = self.model(batch_state, env_id).gather(1, batch_action)
        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states, env_id)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states, env_id).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma * max_next_q_values

        loss = self.MSE(expected_q_values,  current_q_values)
        loss = loss.mean()
        return loss

    def update(self, t_s, t_q, s_s, s_a, s_r, s_s_, env_id, frame):

        self.teacher_memory_list[env_id].push((t_s, t_q))
        self.student_memory_list[env_id].push((s_s, s_a, s_r, s_s_))

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None
        
        batch_vars_kd = self.prep_minibatch_kd(env_id)
        kd_loss = self.compute_loss_kd(batch_vars_kd, env_id)

        batch_vars_dqn = self.prep_minibatch_dqn(env_id)
        dqn_loss = self.compute_loss_dqn(batch_vars_dqn, env_id)

        # print('kd_loss', kd_loss)
        loss = kd_loss*self.kl_ratio + dqn_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1) # adjustable
        self.optimizer.step()

        self.update_target_model()
        self.save_td(loss.item(), frame, env_id)
        self.save_sigma_param_magnitudes(frame, env_id)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass

    def MSE(self, x, y):
        z = (x-y)
        return 0.5 * z.pow(2)
    
    def KL(self, target, input):
        target = F.softmax(target, dim=1)
        # input = input / self.kl_tao
        input = F.softmax(input, dim=1)
        return (target*(target.log() - input.log())).sum()

    def get_max_next_state_action(self, next_states, env_id):
        return self.target_model(next_states, env_id).max(dim=1)[1].view(-1, 1)

    def save_sigma_param_magnitudes(self, tstep, env_id):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_ += torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            if count > 0:
                with open(os.path.join(self.log_dir_list[env_id], 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count))

    def save_td(self, td, tstep, env_id):
        with open(os.path.join(self.log_dir_list[env_id], 'td.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_val_res(self, log_dir, env_name, num_frame, reward):
        with open(osp.join(log_dir, env_name+'val_log.txt'), 'a') as f:
            f.write('{}, {} \n'.format(num_frame, reward))

    def save_w(self, name='model'):
        if not osp.exists('saved_agents'):
            os.mkdir('saved_agents')
        torch.save(self.model.state_dict(), './saved_agents/{}.dump'.format(name))
        torch.save(self.optimizer.state_dict(), './saved_agents/{}_optim.dump'.format(name))

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())