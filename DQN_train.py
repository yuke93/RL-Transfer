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

import matplotlib
# %matplotlib inline
from IPython.display import clear_output

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import *
from utils.hyperparameters import Config
from utils.plot import plot_reward

from multi_agent import Multi_Agent

config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + \
    (config.epsilon_start - config.epsilon_final) * \
    math.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables
config.GAMMA = 0.99
config.LR = 1e-4

# memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 10000  # 100000
config.BATCH_SIZE = 32

# Learning control variables
config.LEARN_START = 500  # 10000
# config.MAX_FRAMES = 500000
# config.MAX_FRAMES = 500000
config.MAX_FRAMES = 2000
config.EACH_MAX_FRAMES = 1000
config.UPDATE_FREQ = 1  # 1

config.loss_type = 'KL'
config.KL_TAO = 0.01
config.KL_ratio = 0.1

# for validation
config.VALID_INTERVAL = 10000
config.VALID_EPISODES = 5


if __name__ == '__main__':

    start = timer()

    valid_env_name_list = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4",
                           "DemonAttackNoFrameskip-v4", "AssaultNoFrameskip-v4"]

    env_names = [int(sys.argv[1]), int(sys.argv[2])]
    env_name_list = [valid_env_name_list[i] for i in env_names]

    log_dir_list = list()
    for env_name in env_name_list:
        print('Init env: ' + env_name)
        log_dir = osp.join('log', env_name)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        log_dir_list.append(log_dir)

    env_list = list()
    for i, env_name in enumerate(env_name_list):
        t_env = make_atari(env_name)
        t_env = wrap_deepmind(t_env, episode_life=True,
                              clip_rewards=True, frame_stack=False, scale=True)
        t_env = WrapPyTorch(t_env)

        s_env = make_atari(env_name)
        log_dir = log_dir_list[i]
        s_env = bench.Monitor(s_env, os.path.join(log_dir, env_name))
        s_env = wrap_deepmind(s_env, episode_life=True,
                              clip_rewards=True, frame_stack=False, scale=True)
        s_env = WrapPyTorch(s_env)
        env_list.append((env_name, t_env, s_env))

    gpu_id = 0
    with torch.cuda.device(gpu_id):
        model = Multi_Agent(env_list=env_list, config=config,
                            log_dir_list=log_dir_list)

        num_frames_list = [0 for i in range(len(env_list))]
        complete = [False for i in range(len(env_list))]
        t_obs_list = [None for i in range(len(env_list))]
        s_obs_list = [None for i in range(len(env_list))]
       
        while(not all(complete)):

            for env_id, (env_name, t_env, s_env) in enumerate(env_list):

                if num_frames_list[env_id] <= 0:
                    t_observation = t_env.reset()
                    s_observation = s_env.reset()
                else:
                    t_observation = t_obs_list[env_id]
                    s_observation = s_obs_list[env_id]

                num_frames = num_frames_list[env_id]
                if num_frames >= config.MAX_FRAMES:
                    complete[env_id] = True
                    continue

                for frame_idx in range(num_frames+1, num_frames + config.EACH_MAX_FRAMES + 1):
                    t_Q, t_action = model.get_Q(t_observation, env_id)
                    prev_t_observation = t_observation
                    t_observation, t_reward, t_done, _ = t_env.step(t_action)
                    t_observation = None if t_done else t_observation

                    epsilon = config.epsilon_by_frame(frame_idx)
                    s_action = model.get_action(s_observation, env_id, epsilon)
                    prev_s_observation = s_observation
                    s_observation, s_reward, s_done, _ = s_env.step(s_action)
                    s_observation = None if s_done else s_observation

                    model.update(prev_t_observation, t_Q, prev_s_observation,
                                 s_action, s_reward, s_observation, env_id, frame_idx)

                    if t_done:
                        t_observation = t_env.reset()
                    
                    if s_done:
                        s_observation = s_env.reset()

                    if frame_idx % 1000 == 0:
                        print('{}, Frame: {}'.format(env_name, frame_idx))
                    
                    if frame_idx % 10000 == 0:
                        try:
                            log_dir = log_dir_list[env_id]
                            save_filename = osp.join(log_dir, env_name+'results.png')
                            clear_output(True)
                            plot_reward(log_dir, env_name, 'DQN', config.MAX_FRAMES, bin_size=10, smooth=1,
                                        time=timedelta(seconds=int(timer()-start)), ipynb=False, save_filename=save_filename)
                        except IOError:
                            pass

                num_frames += config.EACH_MAX_FRAMES

                # complete
                if num_frames >= config.MAX_FRAMES:
                    complete[env_id] = True
                    model.save_w(env_name)
                    t_env.close()
                    s_env.close()

                num_frames_list[env_id] = num_frames
                t_obs_list[env_id] = t_observation
                s_obs_list[env_id] = s_observation