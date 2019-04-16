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
config.MAX_FRAMES = 5000
config.EACH_MAX_FRAMES = 500
config.UPDATE_FREQ = 1  # 1

config.loss_type='KL'
config.KL_TAO = 0.01

# for validation
config.VALID_INTERVAL = 10000
config.VALID_EPISODES = 5


if __name__ == '__main__':

    start = timer()


    valid_env_name_list = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4",
                      "DemonAttackNoFrameskip-v4", "AssaultNoFrameskip-v4"]

    env_names = [0, 3]
    env_name_list = [valid_env_name_list[i] for i in env_names]

    log_dir_list = list()
    student_dir = 'student_'
    student_dir = osp.join('log', student_dir)
    for env_name in env_name_list:
        print('Init env: ' + env_name)
        log_dir = osp.join('log', env_name)
        student_dir += env_name[:3]
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        log_dir_list.append(log_dir)

    # make student dir
    if not osp.exists(student_dir):
        os.makedirs(student_dir)
    
    env_list = list()
    for i, env_name in enumerate(env_name_list):
        env = make_atari(env_name)
        log_dir = log_dir_list[i]
        #env = bench.Monitor(env, os.path.join(log_dir, env_name))
        env = wrap_deepmind(env, episode_life=True,
                            clip_rewards=True, frame_stack=False, scale=True)
        env = WrapPyTorch(env)
        env_list.append( (env_name, env) )

    # make student env
    student_env_list = list()
    for i, env_name in enumerate(env_name_list):
        env = make_atari(env_name)
        env = bench.Monitor(env, os.path.join(student_dir, env_name))
        env = wrap_deepmind(env, episode_life=True,
                            clip_rewards=True, frame_stack=False, scale=True)
        env = WrapPyTorch(env)
        student_env_list.append( (env_name, env) )

    # next validation frame for each game
    next_valid_frames = [config.VALID_INTERVAL] * len(env_names)

    gpu_id = 0
    with torch.cuda.device(gpu_id):
        model = Multi_Agent(env_list=env_list, config=config, log_dir_list=log_dir_list)

        episode_reward_list = [0 for i in range(len(env_list))]
        num_frames_list = [0 for i in range(len(env_list))]
        env_status_list = [None, None]
        complete = [False, False]

        while(not all(complete)):

            for env_id, (env_name, env) in enumerate(env_list):

                if num_frames_list[env_id] <= 0:
                    observation = env.reset()
                else:
                    observation = env_status_list[env_id]
                episode_reward = episode_reward_list[env_id]
                num_frames = num_frames_list[env_id]

                if num_frames >= config.MAX_FRAMES: 
                    complete[env_id] = True
                    continue

                for frame_idx in range(num_frames+1, num_frames + config.EACH_MAX_FRAMES + 1):

                    Q, action = model.get_Q(observation, env_id)
                    prev_observation = observation
                    observation, reward, done, _ = env.step(action)
                    observation = None if done else observation

                    model.update(prev_observation, Q, env_id, frame_idx)
                    episode_reward += reward

                    if done:
                        model.finish_nstep()
                        model.reset_hx()
                        observation = env.reset()
                        model.save_reward(episode_reward)
                        episode_reward = 0

                    if frame_idx % 1000 == 0:
                        print('{}, Frame: {}'.format(env_name, frame_idx))


                num_frames += config.EACH_MAX_FRAMES

                # validation
                if num_frames >= next_valid_frames[env_id]:
                    print('[Validation] env_id: {}, No. frame: {}'.format(env_id, num_frames))
                    num_done = 0
                    valid_env = student_env_list[env_id][1]
                    valid_observation = valid_env.reset()
                    valid_reward_list = []
                    valid_episode_reward = 0.
                    while num_done <= config.VALID_EPISODES:
                        _, valid_action = model.get_policy_Q(valid_observation, env_id)
                        valid_observation, valid_reward, done, _ = valid_env.step(valid_action)
                        valid_episode_reward += valid_reward
                        # print(reward, done)
                        if done:
                            valid_observation = valid_env.reset()
                            num_done += 1
                            valid_reward_list.append(valid_episode_reward)
                            valid_episode_reward = 0.
                    valid_mean_reward = np.asarray(valid_reward_list).mean()
                    model.save_val_res(student_dir, env_name, num_frames, valid_mean_reward)
                    next_valid_frames[env_id] += config.VALID_INTERVAL

                # complete
                if num_frames >= config.MAX_FRAMES:
                    complete[env_id] = True
                    model.save_w(env_name)
                    env.close()

                num_frames_list[env_id] = num_frames
                episode_reward_list[env_id] = episode_reward
                env_status_list[env_id] = observation