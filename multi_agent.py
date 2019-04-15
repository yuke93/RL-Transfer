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
        for env_name, env in env_list:
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

    def __init__(self, env_list, config, log_dir_list):

        super(Multi_Agent, self).__init__()

        self.log_dir_list = log_dir_list
        self.env_list = env_list
        self.num_feats_list = [
            env.observation_space.shape for _, env in env_list]
        self.num_actions_list = [env.action_space.n for _, env in env_list]

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

        self.rewards = list()

        # load teacher model
        self.load_teacher_model()
        # build model
        self.model = DQN_multi(self.env_list)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = self.model.to(self.device)
        for teacher_model in self.teacher_model_list:
            teacher_model = teacher_model.to(self.device)

        self.declare_memory()

    def load_teacher_model(self):
        self.teacher_model_list = list()
        for env_name, env in self.env_list:
            num_feats = env.observation_space.shape
            num_actions = env.action_space.n
            model = DQN(num_feats, num_actions)
            pretrained_weights = osp.join(
                'pretrained_weights', env_name+'.dump')
            model.load_state_dict(torch.load(pretrained_weights))
            for param in model.parameters():
                param.requires_grad = False
            self.teacher_model_list.append(model)

    def declare_memory(self):
        self.memory_list = list()
        for env_name, env in self.env_list:
            memory = ExperienceReplayMemory(self.experience_replay_size)
            self.memory_list.append(memory)

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

    def prep_minibatch(self, env_id):
        # random transition batch is taken from experience replay memory
        transitions = self.memory_list[env_id].sample(self.batch_size)
        batch_state, batch_Q = zip(*transitions)
        shape = (-1,) + self.num_feats_list[env_id]
        batch_state = torch.tensor(
            batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_Q = torch.tensor(
            batch_Q, device=self.device, dtype=torch.float)
        batch_Q = batch_Q.detach()
        batch_Q = torch.squeeze(batch_Q)
        return batch_state, batch_Q

    def compute_loss(self, batch_vars, env_id):
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

    def update(self, s, q, env_id, frame):

        self.memory_list[env_id].push((s, q))

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        # not implemented yet !
        batch_vars = self.prep_minibatch(env_id)
        loss = self.compute_loss(batch_vars, env_id)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1) # adjustable
        self.optimizer.step()

        self.save_td(loss.item(), frame, env_id)
        self.save_sigma_param_magnitudes(frame, env_id)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass

    def save_reward(self, reward):
        self.rewards.append(reward)

    def MSE(self, x, y):
        z = (x-y)
        return 0.5 * z.pow(2)
    
    def KL(self, input, target):
        target = target / self.kl_tao
        target = F.softmax(target, dim=1)
        input = F.softmax(input, dim=1)
        return (target*(target.log() - input.log())).sum()


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