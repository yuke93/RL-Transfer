import os, sys, shutil
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

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import *
from utils.hyperparameters import Config
from utils.plot import plot_reward
from model import Model


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
config.EXP_REPLAY_SIZE = 100000  # 100000
config.BATCH_SIZE = 32

# Learning control variables
config.LEARN_START = 10000  # 10000
config.MAX_FRAMES = 1000000  # 1000000
config.UPDATE_FREQ = 1  # 1


if __name__ == '__main__':

    env_id = sys.argv[1]
    valid_env_list = ["DemonAttackNoFrameskip-v4", "PongNoFrameskip-v4",
                      "AirRaidNoFrameskip-v4", "AssaultNoFrameskip-v4", "CarnivalNoFrameskip-v4"]
    assert env_id in valid_env_list
    print('Init env: ' + env_id)

    # log_dir = "/tmp/gym/"
    log_dir = osp.join('log', env_id)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    env = make_atari(env_id)
    env = bench.Monitor(env, os.path.join(log_dir, env_id))
    env = wrap_deepmind(env, episode_life=True,
                        clip_rewards=True, frame_stack=False, scale=True)
    env = WrapPyTorch(env)

    start = timer()
    gpu_id = 0
    with torch.cuda.device(gpu_id):
        model = Model(env=env, config=config, log_dir=log_dir)

        episode_reward = 0

        observation = env.reset()
        for frame_idx in range(1, config.MAX_FRAMES + 1):
            epsilon = config.epsilon_by_frame(frame_idx)

            action = model.get_action(observation, epsilon)
            prev_observation = observation
            observation, reward, done, _ = env.step(action)
            observation = None if done else observation

            model.update(prev_observation, action,
                         reward, observation, frame_idx)
            episode_reward += reward

            if done:
                model.finish_nstep()
                model.reset_hx()
                observation = env.reset()
                model.save_reward(episode_reward)
                episode_reward = 0

            if frame_idx % 1000 == 0:
                print('Frame: {}'.format(frame_idx))

            if frame_idx % 1000 == 0:
            # if frame_idx % 100 == 0:
                try:
                    save_filename = osp.join(log_dir, env_id, env_id+'.png')
                    print("plot_reward", frame_idx)
                    plot_reward(log_dir, env_id, 'DQN', config.MAX_FRAMES, bin_size=10, smooth=1,
                                time=timedelta(seconds=int(timer()-start)), ipynb=False, save_filename=save_filename)
                except IOError:
                    pass

        model.save_w(env_id)
        # model.save_w()
    env.close()
